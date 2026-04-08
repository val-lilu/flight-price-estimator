import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ===== CONFIG =====
BASE_URL = "https://app.scrapingbee.com/api/v1/"
API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
HEADERS = {"User-Agent": "Mozilla/5.0"}

SWISS_AIRPORTS = ["ZRH"]
TOP_DESTINATIONS = ["CDG"]

START_DATE = datetime(2025, 6, 1)
NUM_DAYS = 92
DATE_LIST = [
    (START_DATE + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range(NUM_DAYS)
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not API_KEY:
    raise ValueError("SCRAPINGBEE_API_KEY environment variable is not set.")

all_flights = []


def build_target_url(origin: str, destination: str, date: str) -> str:
    return (
        f"https://flights.booking.com/flights/{origin}.AIRPORT-{destination}.AIRPORT/"
        f"?type=ONEWAY&adults=1&cabinClass=ECONOMY&children="
        f"&from={origin}.AIRPORT&to={destination}.AIRPORT"
        f"&depart={date}&sort=BEST&travelPurpose=leisure&selected_currency=CHF"
    )


current_date = datetime.now()

for origin in SWISS_AIRPORTS:
    for destination in TOP_DESTINATIONS:
        if origin == destination:
            continue

        for date in DATE_LIST:
            departure_date = datetime.strptime(date, "%Y-%m-%d")
            days_until_flight = (departure_date - current_date).days

            for page in range(1, 2):
                print(f"\n📄 Scraping {origin} ➡️ {destination} | {date} | Page {page}...")

                target_url = f"{build_target_url(origin, destination, date)}&page={page}"
                params = {
                    "api_key": API_KEY,
                    "url": target_url,
                    "render_js": "true",
                    "wait": "6000",
                    "premium_proxy": "true",
                    "block_resources": "false",
                    "country_code": "ch",
                }

                try:
                    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=60)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")

                        price_divs = soup.find_all(
                            "div",
                            attrs={"data-testid": "flight_card_price_main_price"},
                        )

                        airport_blocks = soup.find_all(
                            "div",
                            class_="Text-module__root--variant-small_1___JYtke "
                                  "Text-module__root--color-neutral_alt___aZ0+x "
                                  "styles-module__point__text___SnKTZ",
                        )

                        time_blocks = soup.find_all(
                            "div",
                            class_="Text-module__root--variant-strong_1___41FP0 "
                                  "styles-module__point__text___SnKTZ",
                        )

                        stops_spans = soup.find_all(
                            "span",
                            class_="Badge-module__text___wRZV1",
                        )

                        regex_pattern = r"(?i)(\b\d+\s+stops?\b|\bdirect\b)"
                        valid_stops = [
                            span.text.strip()
                            for span in stops_spans
                            if re.search(regex_pattern, span.text)
                        ]

                        airline_divs = soup.select(
                            "div.Frame-module__flex-wrap_wrap___jtOMN "
                            "div.Text-module__root--variant-small_1___JYtke"
                        )
                        airlines = [
                            div.get_text(strip=True)
                            for div in airline_divs[:len(price_divs)]
                        ]

                        flight_count = min(20, len(price_divs))

                        for i in range(flight_count):
                            price = price_divs[i].get_text(strip=True)
                            num_stops = valid_stops[i] if i < len(valid_stops) else "N/A"

                            dep_code = "N/A"
                            arr_code = "N/A"
                            dep_date = "N/A"
                            arr_date = "N/A"
                            dep_time = "N/A"
                            arr_time = "N/A"

                            if i * 2 + 1 < len(airport_blocks):
                                dep_block = airport_blocks[i * 2]
                                arr_block = airport_blocks[i * 2 + 1]

                                dep_span = dep_block.find("span")
                                arr_span = arr_block.find("span")

                                dep_code = dep_span.get_text(strip=True) if dep_span else "N/A"
                                arr_code = arr_span.get_text(strip=True) if arr_span else "N/A"

                                dep_text = dep_block.get_text(strip=True)
                                arr_text = arr_block.get_text(strip=True)

                                dep_date = dep_text.replace(dep_code, "").strip("· ").strip()
                                arr_date = arr_text.replace(arr_code, "").strip("· ").strip()

                            if i * 2 + 1 < len(time_blocks):
                                dep_time = time_blocks[i * 2].get_text(strip=True)
                                arr_time = time_blocks[i * 2 + 1].get_text(strip=True)

                            airline = airlines[i] if i < len(airlines) else "N/A"

                            all_flights.append({
                                "origin": origin,
                                "destination": destination,
                                "date_queried": date,
                                "departure_airport": dep_code,
                                "departure_time": dep_time,
                                "departure_date": dep_date,
                                "arrival_airport": arr_code,
                                "arrival_time": arr_time,
                                "arrival_date": arr_date,
                                "airline": airline,
                                "number_of_stops": num_stops,
                                "price": price,
                                "days_until_flight": days_until_flight,
                            })

                            print(
                                f"✅ {dep_code} {dep_time} → {arr_code} {arr_time} | "
                                f"✈️ {airline} | Stops: {num_stops} | "
                                f"Days until flight: {days_until_flight} | {price}"
                            )

                    else:
                        print(f"❌ Failed: {response.status_code}")
                        print(response.text[:300])

                    time.sleep(1.5)

                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    time.sleep(2)

output_path = os.path.join(OUTPUT_DIR, "ZRH-CDG.csv")
df = pd.DataFrame(all_flights)
df.to_csv(output_path, index=False)

print(f"\n📁 Saved scraped flights to: {output_path}")
