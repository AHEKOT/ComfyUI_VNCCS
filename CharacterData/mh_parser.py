import os
import numpy as np

class TargetParser:
    def __init__(self, makehuman_path):
        self.makehuman_path = makehuman_path
        self.macro_targets = []
        
        # Define categories to look for in filenames
        self.categories = {
            'gender': ['male', 'female'],
            'age': ['baby', 'child', 'young', 'old'],
            'muscle': ['minmuscle', 'averagemuscle', 'maxmuscle'],
            'weight': ['minweight', 'averageweight', 'maxweight'],
            'height': ['minheight', 'averageheight', 'maxheight'],
            'race': ['african', 'asian', 'caucasian'],
            'cup': ['mincup', 'averagecup', 'maxcup'],
            'firmness': ['minfirmness', 'averagefirmness', 'maxfirmness'],
            'universal': ['universal'], # Special flag
            # Genital specific tags
            'penis_len': ['penis-length-decr', 'penis-length-incr'],
            'penis_circ': ['penis-circ-decr', 'penis-circ-incr'],
            'penis_test': ['penis-testicles-decr', 'penis-testicles-incr'],
        }

    def scan_targets(self):
        """
        Scans macrodetails, breast, and genitals folders.
        """
        base_folders = ["macrodetails", "breast", "genitals"]
        all_targets = []
        
        for folder in base_folders:
            base_path = os.path.join(self.makehuman_path, "makehuman", "data", "targets", folder)
            if not os.path.exists(base_path):
                # Try fallback path if repo structure differs
                base_path = os.path.join(self.makehuman_path, "data", "targets", folder)
                if not os.path.exists(base_path):
                     # print(f"Warning: Target folder not found {folder}")
                     continue

            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith(".target"):
                        full_path = os.path.join(root, file)
                        tags = self._parse_filename(file)
                        
                        # Filtering Logic to prevent unwanted modifiers (like breast-dist, breast-height) 
                        # from being applied essentially with weight 1.0 because they match no categories.
                        keep = False
                        
                        if folder == "macrodetails":
                            # Macrodetails usually define the base. 
                            # Check if it has any relevant tags (race, age, gender, universal)
                            # Actually, we expect ALL macrodetails to be race-based or universal.
                            # If it has NO tags, it's dangerous.
                            if len(tags) > 0:
                                keep = True
                                
                        elif folder == "breast":
                            # Only keep permutation targets (Cup/Firmness)
                            # These files always contain 'cup' and 'firmness' tags.
                            if 'cup' in tags:
                                keep = True
                                
                        elif folder == "genitals":
                            # Only keep supported genital modifiers
                            # Check if any tag starts with 'penis'
                            if any(k.startswith('penis') for k in tags):
                                keep = True
                        
                        if keep:
                            all_targets.append({
                                'path': full_path,
                                'tags': tags,
                                'data': None,
                                'filename': file.replace('.target', '')
                            })
                            # Immediately load data 
                            self.load_target_data(all_targets[-1])

        self.macro_targets = all_targets
        return all_targets

    def _parse_filename(self, filename):
        """
        Extracts tags from filename.
        e.g. "universal-male-young.target" -> {'universal': True, 'gender': 'male', 'age': 'young'}
        """
        name = filename.lower().replace('.target', '')
        # Handle "decr" and "incr" logic specially or as full strings
        # The split might break "penis-length-decr".
        # Let's keep the split but also search for specific full ngrams if needed.
        
        parts = name.replace('_', '-').split('-')
        
        tags = {}
        found_categories = set()

        # Check full name against special keys (like penis-length-decr)
        for cat, values in self.categories.items():
            for val in values:
                if val in name: # substring check for complex names? 
                    # Danger: 'penis-length-decr' contains 'length'
                    # Better to check parts or exact mapping.
                    pass

        for part in parts:
            for cat, values in self.categories.items():
                 # Simple match
                 if part in values:
                    tags[cat] = part
                    found_categories.add(cat)
                    
        # Special handling for composite tags in genitals that might not align with simple split
        if 'penis' in parts:
            if 'length' in parts:
                if 'decr' in parts: tags['penis_len'] = 'penis-length-decr'
                if 'incr' in parts: tags['penis_len'] = 'penis-length-incr'
            if 'circ' in parts:
                if 'decr' in parts: tags['penis_circ'] = 'penis-circ-decr'
                if 'incr' in parts: tags['penis_circ'] = 'penis-circ-incr'
            if 'testicles' in parts:
                 if 'decr' in parts: tags['penis_test'] = 'penis-testicles-decr'
                 if 'incr' in parts: tags['penis_test'] = 'penis-testicles-incr'

        # If 'universal' is in filename, mark it
        if 'universal' in parts:
            tags['universal'] = True
            
        return tags

    def load_target_data(self, target_entry):
        """
        Reads the .target file and returns (indices, deltas).
        """
        if target_entry.get('data') is not None:
            return target_entry['data']
            
        indices = []
        deltas = []
        
        try:
            with open(target_entry['path'], 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            idx = int(parts[0])
                            dx = float(parts[1])
                            dy = float(parts[2])
                            dz = float(parts[3])
                            indices.append(idx)
                            deltas.append([dx, dy, dz])
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error reading {target_entry['path']}: {e}")
            return None

        if len(indices) == 0:
            return None

        data = (np.array(indices, dtype=np.int32), np.array(deltas, dtype=np.float32))
        target_entry['data'] = data # Cache it
        return data

class HumanSolver:
    def __init__(self):
        pass

    def calculate_factors(self, age, gender, weight, muscle, height, breast_size, genital_size):
        """
        Returns a dictionary of factor values based on MakeHuman logic.
        breast_size: 0.0-1.0 (Controls cup size)
        genital_size: 0.0-1.0 (Controls penis length/testicles size)
        """
        factors = {}
        
        # Gender
        factors['male'] = gender
        factors['female'] = 1.0 - gender
        
        # Age
        if age < 0.5:
            factors['old'] = 0.0
            factors['baby'] = max(0.0, 1 - age * 5.333)
            factors['young'] = max(0.0, (age - 0.1875) * 3.2)
            factors['child'] = max(0.0, min(1.0, 5.333 * age) - factors['young'])
        else:
            factors['child'] = 0.0
            factors['baby'] = 0.0
            factors['old'] = max(0.0, age * 2 - 1)
            factors['young'] = 1 - factors['old']

        # Muscle
        factors['maxmuscle'] = max(0.0, muscle * 2 - 1)
        factors['minmuscle'] = max(0.0, 1 - muscle * 2)
        factors['averagemuscle'] = 1 - (factors['maxmuscle'] + factors['minmuscle'])

        # Weight
        factors['maxweight'] = max(0.0, weight * 2 - 1)
        factors['minweight'] = max(0.0, 1 - weight * 2)
        factors['averageweight'] = 1 - (factors['maxweight'] + factors['minweight'])

        # Height
        factors['maxheight'] = max(0.0, height * 2 - 1)
        factors['minheight'] = max(0.0, 1 - height * 2)
        factors['averageheight'] = 1 - (factors['maxheight'] + factors['minheight'])

        # Race (Internal Defaults)
        # Even though user input is removed, we must handle the race modifiers in the target files.
        # Defaulting to an even split to produce a generic human base.
        # If we don't set these, targets with 'african'/'asian'/'caucasian' tags will have weight 0.0 (if get returns 0)
        # OR if we removed them from categories, they would have been ignored and thus weight 1.0 (collision).
        # Since we added them back to categories, we must provide non-zero weights if we want them to show.
        factors['african'] = 0.333
        factors['asian'] = 0.333
        factors['caucasian'] = 0.334

        # Breast Size (Cup)
        # Using same logic as Weight/Muscle for consistency
        factors['maxcup'] = max(0.0, breast_size * 2 - 1)
        factors['mincup'] = max(0.0, 1 - breast_size * 2)
        factors['averagecup'] = 1 - (factors['maxcup'] + factors['mincup'])
        
        # Firmness (Hardcoded to average for now, as user didn't request a slider)
        factors['maxfirmness'] = 0.0
        factors['minfirmness'] = 0.0
        factors['averagefirmness'] = 1.0

        # Genitals (Penis Length & Testicles)
        # 0.0 -> max decrease
        # 0.5 -> neutral
        # 1.0 -> max increase
        factors['penis-length-incr'] = max(0.0, genital_size * 2 - 1)
        factors['penis-length-decr'] = max(0.0, 1 - genital_size * 2)
        
        # Mapping penis testicles to same slider
        factors['penis-testicles-incr'] = factors['penis-length-incr']
        factors['penis-testicles-decr'] = factors['penis-length-decr']
        
        # Circumference not explicitly requested, keeping neutral
        factors['penis-circ-incr'] = 0.0
        factors['penis-circ-decr'] = 0.0

        # Universal
        factors['universal'] = 1.0 # Always true if present

        return factors

    def solve_mesh(self, base_mesh, targets, factors):
        """
        Applies targets to base mesh based on factors.
        Returns a NEW numpy array of vertices.
        """
        new_verts = base_mesh.vertices.copy()
        
        for target in targets:
            # Calculate weight for this target
            weight = 1.0
            relevant = True
            
            # Check all tags in target
            # e.g. {'gender': 'male', 'age': 'young'}
            # Weight = factors['male'] * factors['young']
            
            for cat, tag_val in target['tags'].items():
                if tag_val is True: # e.g. universal=True
                    val = factors.get('universal', 1.0)
                else:
                    val = factors.get(tag_val, 0.0) # If tag not in factors (e.g. proportions), weight is 0
                
                weight *= val
                if weight < 0.001:
                    relevant = False
                    break
            
            if relevant:
                # Load data if not loaded
                # We need the parser instance or the data must be pre-loaded
                # Assuming 'target' is the dictionary from TargetParser
                if target['data'] is None:
                    # Look, ideally we passed the parser or loaded it. 
                    # For now assume data is loaded or we skip.
                    # Wait, I put load_target_data in Parser.
                    pass
                
                if target['data'] is not None:
                    indices, deltas = target['data']
                    # Apply
                    # new_verts[indices] += deltas * weight
                    # Numpy advanced indexing
                    new_verts[indices] += deltas * weight
        
        return new_verts
