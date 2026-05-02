import torch
from env import MegaMekEnvironment

def test_parsing():
    MegaMekEnvironment.connect = lambda self: None
    env = MegaMekEnvironment()
    
    # Test WEAPON_BC
    payload_weapon = {
        "context": "WEAPON_BC",
        "mask": {
            "valid_twists": [
                {
                    "twist": 0,
                    "valid_targets": [
                        {
                            "target_entity_index": 1,
                            "valid_weapons": [
                                {"weapon_id": 5, "to_hit": 7},
                                {"weapon_id": 6, "to_hit": 8}
                            ]
                        }
                    ]
                }
            ]
        },
        "target_action": {
            "torso_twist": 0,
            "attacks": [
                {"target_entity_index": 1, "weapon_id": 5}
            ]
        }
    }
    
    # Mocking board and entities so env doesn't crash on extract
    data_w, _ = env._parse_to_heterodata(payload_weapon)
    
    print("\n--- WEAPON_BC Result ---")
    print(f"y_sequence: {getattr(data_w, 'y_sequence', None)}")
    print(f"Action features: {data_w['action'].x}")
    print(f"Action sources: {data_w['action'].source_unit_idx}")
    print(f"Action target units: {data_w['action'].target_unit_idx}")
    print(f"Action target weapons: {data_w['action'].target_weapon_idx}")
    
    
    # Test PHYSICAL_BC
    payload_physical = {
        "context": "PHYSICAL_BC",
        "mask": {
            "valid_targets": [
                {
                    "target_entity_index": 1,
                    "valid_attacks": [
                        {"action_type": 4, "to_hit": 5}, # KICK_LEFT
                        {"action_type": 5, "to_hit": 6}  # KICK_RIGHT
                    ]
                }
            ]
        },
        "target_action": {
            "attacks": [
                {"target_entity_index": 1, "physical_action_type": 4}
            ]
        }
    }
    
    data_p, _ = env._parse_to_heterodata(payload_physical)
    print("\n--- PHYSICAL_BC Result ---")
    print(f"y_sequence: {getattr(data_p, 'y_sequence', None)}")
    print(f"Action features: {data_p['action'].x}")
    print(f"Action sources: {data_p['action'].source_unit_idx}")
    print(f"Action target units: {data_p['action'].target_unit_idx}")
    print(f"Action target weapons: {data_p['action'].target_weapon_idx}")

if __name__ == "__main__":
    test_parsing()
