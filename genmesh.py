import math
import yaml

def generate_nodes():
    data = {}

    # Node 0 at origin
    data[0] = {
        "antennaGain": 2.0,
        "hopLimit": 7,
        "isClientMute": False,
        "isRepeater": False,
        "isRouter": False,
        "neighborInfo": False,
        "x": 0.00,
        "y": 0.00,
        "z": 1.0
    }

    node_id = 1

    # Generate rings 1–8
    for k in range(1, 9):
        radius = 5000 * k
        count = 8 * k
        angle_step = 2 * math.pi / count

        for i in range(count):
            theta = i * angle_step
            x = round(radius * math.cos(theta), 2)
            y = round(radius * math.sin(theta), 2)

            data[node_id] = {
                "antennaGain": 2.0,
                "hopLimit": 7,
                "isClientMute": False,
                "isRepeater": False,
                "isRouter": False,
                "neighborInfo": False,
                "x": x,
                "y": y,
                "z": 1.0
            }

            node_id += 1

    return data


# Generate data
nodes = generate_nodes()

# Write to YAML file
with open("nodes.yaml", "w") as f:
    yaml.dump(nodes, f, sort_keys=True)

print("Finished writing nodes.yaml with 289 nodes!")
