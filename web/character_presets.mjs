const FIELD_CATEGORIES = {
    race: ["races"],
    skin_color: ["skin_color"],
    body: ["body_type", "breast_size"],
    face: ["face_shape", "face_details"],
    hair: ["hair_color", "hair_pattern", "hair_length", "hair_texture", "hairstyles", "hair_framing"],
    eyes: ["eye_color", "eye_features"],
    additional_details: ["details"],
};
const LEGACY_FIELD_CATEGORIES = {
    hair: ["hair_color", "hairstyles"],
    eyes: ["eyes.colors", "eyes.features"],
    face: ["eyes.face_characteristics"],
};

export function presetGroups(catalog, field) {
    const groups = [];
    const categories = catalog?.tags?.eyes ? LEGACY_FIELD_CATEGORIES[field] || FIELD_CATEGORIES[field] : FIELD_CATEGORIES[field];
    for (const category of categories || []) {
        const byGroup = new Map();
        const items = category.split(".").reduce((value, key) => value?.[key], catalog?.tags);
        for (const item of items || []) {
            const header = item.group || category.replaceAll(".", " ").replaceAll("_", " ");
            if (!byGroup.has(header)) byGroup.set(header, []);
            byGroup.get(header).push(item);
        }
        for (const [header, items] of byGroup) groups.push({ header, items });
    }
    return groups;
}

const presetKey = value => String(value).replaceAll("_", " ").toLowerCase().trim().replace(/\s+/g, " ");

export function presetSelection(value, groups) {
    const aliases = new Map();
    for (const { items } of groups) {
        for (const item of items) {
            for (const alias of [item.tag, item.label, ...(item.synonyms || [])]) {
                aliases.set(presetKey(alias), presetKey(item.tag));
            }
        }
    }
    // Preserve custom text and legacy spelling until the user toggles that choice.
    const selected = new Map();
    for (const token of String(value || "").split(",").map(part => part.trim()).filter(Boolean)) {
        const key = aliases.get(presetKey(token)) || presetKey(token);
        if (!selected.has(key)) selected.set(key, token);
    }
    // The old creation route stored both default hair traits in one token.
    const legacyHair = [aliases.get("black hair"), aliases.get("long hair")].filter(Boolean);
    const hasLegacyHair = key => legacyHair.length === 2 && selected.has("black long hair") && legacyHair.includes(key);
    return {
        has: item => selected.has(presetKey(item.tag)) || hasLegacyHair(presetKey(item.tag)),
        toggle(item) {
            const key = presetKey(item.tag);
            if (hasLegacyHair(key)) {
                selected.delete("black long hair");
                for (const trait of legacyHair) {
                    if (!selected.has(trait)) selected.set(trait, trait);
                }
            }
            if (selected.has(key)) selected.delete(key);
            else selected.set(key, item.tag.replaceAll("_", " "));
        },
        value: () => Array.from(selected.values()).join(", "),
    };
}
