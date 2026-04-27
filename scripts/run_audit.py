from hackathon_opti.data import load_raw_dataset, save_processed_artifacts


if __name__ == "__main__":
    bundle = load_raw_dataset()
    result = save_processed_artifacts(bundle)
    print("Saved processed artifacts:")
    for key, value in result.items():
        print(f"- {key}: {value}")
