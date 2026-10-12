"""nyload-data: download raw NYISO/weather data and build processed parquet files.

    nyload-data all --years 2013-2023
    nyload-data download --years 2020 --weather-source weather_com --station KLGA
    nyload-data build --years 2020-2021
"""
import argparse

from nyload import paths
from nyload.data import build


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        years.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(years))


def main(argv=None) -> None:
    paths.load_dotenv()
    parser = argparse.ArgumentParser(prog="nyload-data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "build", "all"])
    parser.add_argument("--years", type=parse_years, required=True, help="e.g. 2013-2023 or 2019,2021")
    parser.add_argument("--zone", default="CAPITL")
    parser.add_argument("--weather-source", default="open_meteo", choices=["open_meteo", "weather_com"])
    parser.add_argument("--station", default="KALB", help="KALB (Albany, CAPITL zone) or KLGA (NYC)")
    parser.add_argument("--no-forecast", action="store_true", help="skip the NYISO forecast benchmark")
    args = parser.parse_args(argv)

    print(f"data dir: {paths.data_dir()}")
    if args.command in ("download", "all"):
        build.download(args.years, args.weather_source, args.station, forecast=not args.no_forecast)
    if args.command in ("build", "all"):
        build.build(args.years, args.zone, args.weather_source, args.station, forecast=not args.no_forecast)


if __name__ == "__main__":
    main()
