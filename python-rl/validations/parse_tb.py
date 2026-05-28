from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob

log_dir = "runs/impala_master"
event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

# Sort by modification time
event_files.sort(key=os.path.getmtime)

print(f"Found {len(event_files)} event files.")
for f in event_files[-4:]:
    print(f"Reading {f}")
    acc = EventAccumulator(f)
    acc.Reload()
    tags = acc.Tags()['scalars']
    for tag in tags:
        events = acc.Scalars(tag)
        print(f"  {tag}: {len(events)} events. Last value: {events[-1].value} at step {events[-1].step}")

