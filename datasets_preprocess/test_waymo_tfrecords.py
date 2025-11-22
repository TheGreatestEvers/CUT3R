import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import scenario_pb2  # <- this is key

tfrecord_path = "/workspace/Waymo/waymo_validation_converted/waymo_validation_converted866c.tfrecord"

dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

for i, data in enumerate(dataset):
    raw = data.numpy()

    # 1) First see what happens with Frame
    frame = open_dataset.Frame()
    try:
        frame.ParseFromString(raw)
        print("Parsed as Frame successfully!")
        print("  num images:", len(frame.images))
        print("  num lasers:", len(frame.lasers))
    except Exception as e:
        print("Failed to parse as Frame:", repr(e))

    # 2) Now try Scenario (Motion dataset)
    scenario = scenario_pb2.Scenario()
    try:
        scenario.ParseFromString(raw)
        print("Parsed as Scenario successfully!")
        print("  scenario_id:", scenario.scenario_id)
    except Exception as e:
        print("Failed to parse as Scenario:", repr(e))

    break  # just first record