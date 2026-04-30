import torch

def analyze_dataset(file_path="bc_dataset_master.pt"):
    try:
        dataset = torch.load(file_path, weights_only=False)
        print(f"Loaded dataset with {len(dataset)} graphs.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    context_counts = {}
    for i, data in enumerate(dataset):
        context = getattr(data, "context", "UNKNOWN")
        context_counts[context] = context_counts.get(context, 0) + 1
        
        # Print first graph of each context
        if context_counts[context] == 1:
            print(f"\n--- First Example of {context} ---")
            print(f"Graph Keys: {data.keys()}")
            print(f"Action Nodes shape: {data['action'].x.shape}")
            if hasattr(data['action'], 'type_flags'):
                print(f"Action type flags: {data['action'].type_flags}")
            if hasattr(data, 'y_sequence'):
                print(f"y_sequence: {data.y_sequence}")
            if hasattr(data['action'], 'source_unit_idx'):
                print(f"source_unit_idx: {data['action'].source_unit_idx}")

    print("\n--- Summary ---")
    for ctx, count in context_counts.items():
        print(f"{ctx}: {count} graphs")

if __name__ == "__main__":
    analyze_dataset()
