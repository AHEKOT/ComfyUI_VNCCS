# This file contains the full set of VNCCS loop control nodes, supporting dynamic inputs
# and value accumulation across loop iterations, modeled after the successful external structure.

print("[VNCCS NODE LOAD] emotion_preview_loop_manager.py is executing.")

# --- Guaranteed Execution Class ---

# Base class to force the node to run on every execution, defeating the cache.
class RandomGuaranteedClass:
    OUTPUT_NODE = True
    RESULT_NODE = True
    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        # Returning NaN forces a recalculation every time, defeating the cache.
        return float("NaN")

# --- Dynamic Input Helper ---
# Used by VNCCS For Loop (Start) and End to dynamically generate pins.
DYNAMIC_VALUE_TYPES = ("INT", "FLOAT", "STRING", "LIST", "IMAGE", "LATENT", "MODEL", "CLIP", "VAE")
MAX_DYNAMIC_VALUES = 16

# Alias for the flow pin type, ensuring it is always a connection pin.
FLOW_TYPE = ("*", {"forceInput": True, "tooltip": "Loop execution flow."})

def generate_hidden_dynamic_input_types(prefix, start_index=2):
    """
    Generates dynamic input types for pins 2 through 16. 
    These pins are truly optional (hidden) as they lack a default value and use forceInput.
    """
    types = {}
    for i in range(start_index, MAX_DYNAMIC_VALUES + 1):
        key = f"{prefix}_{i}"
        # CRITICAL: No default value + optional + forceInput=True ensures the pin is HIDDEN until connected.
        types[key] = (DYNAMIC_VALUE_TYPES, {"optional": True, "forceInput": True})
            
    return types

# --- Node 1: VNCCS For Loop (Start) ---

class VNCCSForLoopStart(RandomGuaranteedClass):
    """
    VNCCS For Loop (Start)
    Manages iteration (INDEX) and forwards initial accumulator values.
    Dynamically creates initial_value_X inputs and value_X_iter outputs.
    """
    
    CATEGORY = "VNCCS/Control Flow"
    FUNCTION = "generate"
    
    # GUARANTEED OUTPUTS: Using strict names FLOW and INDEX to match the required structure, 
    # and defining them here forces them to appear when the node is added.
    RETURN_TYPES = ("*", "INT") 
    RETURN_NAMES = ("FLOW", "INDEX")

    def __init__(self):
        self.iterator = None
        self.end = 0
        self.start = 0
        self.step = 1

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "count": ("INT", { "default": 1, "min": 0, "max": (2**63-1), "step": 1, "display": "number" }),
            },
            "optional": {
                # Input flow from the End node - required connection
                "flow": FLOW_TYPE, 
                
                # Pin 1: VISIBLE pin. CRITICAL: Added forceInput: True to prevent it from becoming an editable field.
                "initial_value_1": (DYNAMIC_VALUE_TYPES, {"optional": True, "forceInput": True}),
            }
        }
        # Add dynamic inputs 2 through 16 to 'optional' (these will be HIDDEN).
        inputs["optional"].update(generate_hidden_dynamic_input_types("initial_value", start_index=2))
        return inputs

    @classmethod
    def get_output_types(cls, *args, **kwargs):
        """
        Dynamically determine *additional* output pins based on connected initial_value inputs.
        Pins 1 and 2 (FLOW, INDEX) are static and defined in RETURN_TYPES/RETURN_NAMES.
        This function only adds the dynamic value_X_iter outputs starting from pin 1.
        """
        output_types = []
        output_names = []
        
        # Check all possible dynamic inputs (initial_value_1, initial_value_2, etc.)
        for i in range(1, MAX_DYNAMIC_VALUES + 1):
            input_name = f"initial_value_{i}"
            
            # The output pin appears ONLY if the corresponding input pin is connected (present in kwargs).
            if input_name in kwargs:
                output_types.append(("*",))
                output_names.append(f"value_{i}_iter")
        
        # NOTE: Returning these as 'optional' correctly appends them to the mandatory RETURN_TYPES (FLOW, INDEX)
        return {"optional": tuple(output_types), "names": tuple(output_names)}
    
    def generate(self, count, flow=None, **kwargs): 
        
        # --- 1. Core Iteration Logic ---
        start = 0 
        step = 1 # Hardcoded step
        
        if self.iterator is None or flow is None:
            if step <= 0: 
                raise ValueError("Step must be 1 or greater.")
            self.start = start
            self.step = step
            self.end = start + count * step
            self.iterator = iter(range(self.start, self.end, self.step))

        try:
            current_index = next(self.iterator)
            print(f"[VNCCS For Loop Start] Running iteration: INDEX={current_index}")
        except StopIteration:
            # Loop is finished. This state is required for the scheduler to stop re-queuing.
            self.iterator = iter(range(self.start, self.end, self.step))
            current_index = next(self.iterator)
        
        # --- 2. Dynamic Value Forwarding ---
        
        # The first two outputs correspond to the mandatory RETURN_TYPES: FLOW and INDEX
        final_outputs = [current_index, current_index] # [FLOW, INDEX]
        
        # Forward initial_value_X as value_X_iter (appended to mandatory outputs)
        for i in range(1, MAX_DYNAMIC_VALUES + 1):
            input_name = f"initial_value_{i}"
            
            # If the input was connected, its value will be in kwargs
            if input_name in kwargs:
                final_outputs.append(kwargs[input_name])
                
        return tuple(final_outputs)
    
    # Register the dynamic output method
    @classmethod
    def IS_OUTPUT_DYNAMIC(cls):
        return True
    
# --- Node 2: VNCCS For Loop (End) ---

class VNCCSForLoopEnd:
    """
    VNCCS For Loop (End)
    Receives the loop signal and the final accumulated values.
    Dynamically creates final_value_X outputs, representing the data accumulated across the loop.
    """
    
    # GUARANTEED OUTPUT: The first output is mandatory and named final_value1.
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("final_value1",)

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                # Connects to For Loop Start's FLOW output.
                "flow": FLOW_TYPE, 
            },
            "optional": {
                 # Pin 1: VISIBLE pin. CRITICAL: Added forceInput: True to prevent it from becoming an editable field.
                "value_1": (DYNAMIC_VALUE_TYPES, {"optional": True, "forceInput": True}),
            }
        }
        # Add dynamic inputs 2 through 16 to 'optional' (these will be HIDDEN).
        inputs["optional"].update(generate_hidden_dynamic_input_types("value", start_index=2))
        return inputs
    
    @classmethod
    def get_output_types(cls, *args, **kwargs):
        """Dynamically appends outputs from final_value2 to final_value16."""
        output_types = []
        output_names = []
        
        # Start from value_2, as final_value1 is already defined in RETURN_TYPES
        for i in range(2, MAX_DYNAMIC_VALUES + 1):
            input_name = f"value_{i}"
            # IMPORTANT: The output pin should only appear if the INPUT pin (value_i) was connected.
            if input_name in kwargs:
                output_types.append(("*",))
                # Uses the clean 'final_valueX' name for the output
                output_names.append(f"final_value{i}") 
        
        # NOTE: Returning as 'optional' correctly appends them to the mandatory RETURN_TYPES (final_value1)
        return {"optional": tuple(output_types), "names": tuple(output_names)}

    FUNCTION = "generate"
    CATEGORY = "VNCCS/Control Flow"

    def generate(self, flow, **kwargs):
        
        # Output the values in order (value1 first)
        outputs = []
        
        for i in range(1, MAX_DYNAMIC_VALUES + 1):
            input_name = f"value_{i}"
            if input_name in kwargs:
                outputs.append(kwargs[input_name])
        
        return tuple(outputs)
    
    @classmethod
    def IS_OUTPUT_DYNAMIC(cls):
        return True


# --- Node 3: The List Extractor Utility (Preserved) ---

class VNCCSListItemExtractor:
    """
    VNCCS List Item Extractor
    Extracts a single item (STRING) from an input list (LIST) at a given index (INT).
    This node now connects to the INDEX output of the VNCCS For Loop (Start) node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Index is editable when unconnected, connectable when connected.
                "index": ("INT", { "default": 0, "min": 0, "step": 1, "display": "number" }),
                # CORRECTED: Explicitly set the type to LIST to ensure list connections work reliably.
                "list_in": ("LIST", {"forceInput": True}), 
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("item_out",)
    FUNCTION = "generate"
    CATEGORY = "VNCCS/Utility"
    
    def generate(self, index, list_in):
        
        if not isinstance(list_in, list):
            # If a single item (not a list) is passed, just return it as a string
            return (str(list_in),) 
        
        if 0 <= index < len(list_in):
            item = list_in[index]
            #print(f"[VNCCS List Item Extractor] Extracted item at index {index}: {item}")
            return (str(item),)
        else:
            error_msg = f"Index {index} is out of bounds for list of size {len(list_in)}."
            print(f"[VNCCS List Item Extractor] ERROR: {error_msg}")
            return ("ERROR_INDEX_OUT_OF_BOUNDS",)

# --- Node Registration ---

NODE_CLASS_MAPPINGS  = {
    "VNCCSForLoopStart": VNCCSForLoopStart,
    "VNCCSForLoopEnd": VNCCSForLoopEnd,
    "VNCCSListItemExtractor": VNCCSListItemExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSForLoopStart": "VNCCS For-Loop-Start",
    "VNCCSForLoopEnd": "VNCCS For-Loop-End",
    "VNCCSListItemExtractor": "VNCCS List Item Extractor",
}
