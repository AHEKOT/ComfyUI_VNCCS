(()=>{
  const TAG='[VNCCS autofill]';
  const DEBUG=false;
  function log(){ if(DEBUG) console.log(TAG, ...arguments); }
  function warn(){ console.warn(TAG, ...arguments); }
  async function fetchConfig(name){
    if(!name||name==='None') {
      log('fetchConfig skipped for', name);
      return null;
    }
    log('fetchConfig started for', name);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const r = await fetch(`/vnccs/config?name=${encodeURIComponent(name)}`, {cache:'no-store', signal: controller.signal});
      clearTimeout(timeoutId);
      if(r.ok){ 
        log('fetched via endpoint');
        return await r.json(); 
      } else {
        log('endpoint status', r.status);
      }
    } catch(e){ 
      if(e.name === 'AbortError') {
        warn('fetch timeout for', name);
      } else {
        warn('endpoint error', e); 
      }
    }
    // Fallbacks
    const encName = encodeURIComponent(name);
    const rawPath = `VN_CharacterCreatorSuit/${name}/${name}_config.json`;
    const candidates = [
      `/output/VN_CharacterCreatorSuit/${encName}/${encName}_config.json`,
      `/view?filename=${encodeURIComponent(rawPath)}&type=output`,
    ];
    for(const url of candidates){
      try{ 
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const r = await fetch(url,{cache:'no-store', signal: controller.signal});
        clearTimeout(timeoutId);
        if(r.ok){ 
          log('fetched via', url); 
          return await r.json(); 
        } else { 
          log('attempt', url, r.status); 
        }
      } catch(e){ 
        if(e.name === 'AbortError') {
          warn('fallback fetch timeout for', url);
        } else {
          warn('error', url, e); 
        }
      }
    }
    warn('all fetch attempts failed for', name);
    return null;
  }
  function apply(node,cfg){
    if(!cfg) return; const ci=cfg.character_info||{}; const map={character_name:ci.name,background_color:ci.background_color,aesthetics:ci.aesthetics,gender:ci.gender,age:ci.age,race:ci.race,eyes:ci.eyes,hair:ci.hair,face:ci.face,body:ci.body,skin_color:ci.skin_color,additional_details:ci.additional_details,negative_prompt:ci.negative_prompt,lora_prompt:ci.lora_prompt,seed:ci.seed};
    (node.widgets||[]).forEach(w=>{ if(map.hasOwnProperty(w.name) && map[w.name]!==undefined && w.value!==map[w.name]){ log('set',w.name); w.value=map[w.name]; if(typeof w.callback==='function'){ try{ w.callback(w.value);}catch(e){} } } });
    try{ node.setDirtyCanvas(true,true);}catch(_){ }
  }
  async function createNewCharacter(name){
    if(!name) return {error:'empty name'};
    try{ const r=await fetch(`/vnccs/create?name=${encodeURIComponent(name)}`); if(!r.ok) return {error:'http '+r.status}; return await r.json(); }catch(e){ return {error:String(e)}; }
  }
  function addCreateButton(node){
  if(node.widgets?.find(w=>w._vnccsCreate)) return;
    const nameWidget = node.widgets?.find(w=>w.name==='character_name');
  function refreshExistingList(node, newName){
      const sel = node.widgets?.find(w=>w.name==='existing_character');
      if(!sel) { warn('no existing_character to refresh'); return; }
      log('OLD refreshExistingList called with newName:', newName, 'type:', typeof newName);
      let list = sel.options || sel.values || sel.choices || sel.items;
      log('OLD list before update:', list);
      if(Array.isArray(list)){
        if(!list.includes(newName)){
          list.push(newName);
          log('OLD list after push:', list);
          if(sel.options) sel.options = list; if(sel.values) sel.values = list; if(sel.choices) sel.choices = list; if(sel.items) sel.items = list;
        }
      }
      sel.value = newName;
      log('OLD set sel.value to:', newName);
      // if(typeof sel.callback === 'function'){ try { sel.callback(newName); } catch(e){} }
      try { node.setDirtyCanvas(true,true); } catch(_){}
    }
    const callback = async ()=>{
      const newName = (nameWidget?.value||'').trim();
      log('nameWidget found, raw value:', nameWidget?.value, 'trimmed:', newName);
      if (!newName) {
        log('Empty character name in new_character_name field');
        return;
      }
      log('create request', newName);
      const res = await createNewCharacter(newName);
      if(res && !res.error){
        log('created', newName, res.existing ? '(existing)' : '');
        refreshExistingList(newName);
        const cfg = await fetchConfig(newName);
        apply(node, cfg);
      } else {
        warn('create failed', res && res.error);
      }
    };
    const w = node.addWidget('button','Create New Character','', callback);
    if(w){
      w._vnccsCreate = true;
      if(nameWidget){
        const list = node.widgets;
        const nameIdx = list.indexOf(nameWidget);
        const btnIdx = list.indexOf(w);
        if(nameIdx >=0 && btnIdx > nameIdx+1){
          list.splice(btnIdx,1);
          list.splice(nameIdx+1,0,w);
        }
      }
      try { node.setDirtyCanvas(true,true); } catch(_){}
    }
  }
  function addCreateButtonSafely(node) {
    if (node.widgets?.find(w => w._vnccsCreateButton)) return;

    log('Adding create button safely to node', node.id);

    function refreshExistingList(newName) {
      const existingWidget = node.widgets?.find(w => w.name === 'existing_character');
      if (!existingWidget) return;

      log('refreshExistingList called with newName:', newName, 'type:', typeof newName);

      const props = ['options', 'values', 'choices', 'items'];
      for (const prop of props) {
        if (Array.isArray(existingWidget[prop])) {
          log('Before update, existingWidget[prop]:', existingWidget[prop]);
          if (!existingWidget[prop].includes(newName)) {
            existingWidget[prop].push(newName);
            log('After push, existingWidget[prop]:', existingWidget[prop]);
          } else {
            log('newName already in list');
          }
        }
      }

      existingWidget.value = newName;
      log('Set existingWidget.value to:', newName);
      try { node.setDirtyCanvas(true,true); } catch(_){}
    }

    const createCallback = async () => {
      log('Available widgets in node', node.id + ':');
      node.widgets?.forEach((w, index) => {
        log(`Widget ${index}: name='${w.name}', type='${w.type}', value='${w.value}'`);
      });

      const nameWidget = node.widgets?.find(w => w.name === 'new_character_name');
      if (!nameWidget) {
        warn('new_character_name widget not found when creating character');
        const widgetNames = node.widgets?.map(w => w.name) || [];
        log('Available widget names:', widgetNames);
        return;
      }

      log('=== WIDGET DEBUG INFO ===');
      log('Widget name:', nameWidget.name);
      log('Widget type:', nameWidget.type);
      log('Widget value (raw):', nameWidget.value);
      log('Widget value type:', typeof nameWidget.value);
      log('Widget hasOwnProperty value:', nameWidget.hasOwnProperty('value'));
      log('Widget options:', nameWidget.options);
      log('Widget default value:', nameWidget.options?.default);
      log('Widget is visible:', nameWidget.visible);
      log('Widget is disabled:', nameWidget.disabled);
      log('Widget callback exists:', typeof nameWidget.callback === 'function');
      log('========================');

      const newName = (nameWidget?.value || '').trim();
      log('Final processed value:', newName, 'length:', newName.length);
      if (!newName) {
        log('Empty character name in new_character_name field');
        return;
      }

      log('create request', newName);
      const res = await createNewCharacter(newName);
      if(res && !res.error){
        log('created', newName, res.existing ? '(existing)' : '');
        refreshExistingList(newName);
        const cfg = await fetchConfig(newName);
        apply(node, cfg);
      } else {
        warn('create failed', res && res.error);
      }
    };

    const button = node.addWidget('button', 'Create New Character', '', createCallback);

    if (button) {
      button._vnccsCreateButton = true;


      try {
        node.setDirtyCanvas?.(true, true);
      } catch (e) {
        warn('Error updating canvas:', e);
      }

      log('Create button added successfully at the end of widget list');
    }
  }
  function applyCostumeData(node, cfg, costume){
    log('applyCostumeData called for node', node.id, 'costume:', costume);
    if(!cfg || !cfg.costumes || !cfg.costumes[costume]) {
      warn('no costume data for', costume);
      return;
    }
    const costumeData = cfg.costumes[costume];
    log('costumeData:', costumeData);
    const map = {
      face: costumeData.face,
      head: costumeData.head,
      top: costumeData.top,
      bottom: costumeData.bottom,
      shoes: costumeData.shoes,
      negative_prompt: costumeData.negative_prompt,
      extra_negative_prompt: costumeData.extra_negative_prompt
    };
    log('fields to update:', map);
    (node.widgets||[]).forEach(w => {
      if(map.hasOwnProperty(w.name) && map[w.name] !== undefined && w.value !== map[w.name]){
        log('updating field', w.name, 'from', w.value, 'to', map[w.name]);
        const tempCallback = w.callback;
        w.callback = null;
        w.value = map[w.name];
        w.callback = tempCallback;
        log('field', w.name, 'updated');
      } else {
        log('field', w.name, 'skipped (no change or not in map)');
      }
    });
    log('applyCostumeData completed without setDirtyCanvas');
  }

  function updateCostumeList(node, cfg){
    log('updateCostumeList called for node', node.id);
    const costumeWidget = node.widgets?.find(w => w.name === 'costume');
    if(!costumeWidget) {
      warn('costumeWidget not found');
      return;
    }
    let costumes = ['Naked'];
    if(cfg && cfg.costumes){
      Object.keys(cfg.costumes).forEach(c => {
        if(!costumes.includes(c)) costumes.push(c);
      });
    }
    log('new costume list:', costumes);
    const oldValue = costumeWidget.value;
    log('old costume value:', oldValue);
    const tempCallback = costumeWidget.callback;
    costumeWidget.callback = null;
    if (!costumeWidget.options) costumeWidget.options = {};
    costumeWidget.options.values = costumes;
    costumeWidget.values = costumes;
    costumeWidget.choices = costumes;
    if (costumeWidget.items) costumeWidget.items = costumes;
    if (!costumes.includes(oldValue)) {
      costumeWidget.value = 'Naked';
      log('set costume to Naked (old not in list)');
    } else {
      log('kept old costume value:', oldValue);
    }
    costumeWidget.callback = tempCallback;
    setTimeout(() => {
      try { node.setDirtyCanvas(true, true); } catch(_) {}
      log('setDirtyCanvas called after updateCostumeList');
    }, 10);
    log('updateCostumeList completed');
  }

  function hookCharacterAssetSelector(node){
    log('hookCharacterAssetSelector called for node', node.id);
    if(node.__vnccsAssetHooked) {
      log('already hooked, skipping');
      return;
    }
    node.__vnccsAssetHooked = true;
    log('hooking CharacterAssetSelector', node.id);

    const charWidget = node.widgets?.find(w => w.name === 'character');
    const costumeWidget = node.widgets?.find(w => w.name === 'costume');
    if(!charWidget || !costumeWidget) {
      warn('character or costume widget not found');
      return;
    }
    log('widgets found: character and costume');

    if(!node.widgets.find(w => w._vnccsCreateCostume)) {
      log('Adding create costume button');
      const nameWidget = node.widgets?.find(w => w.name === 'new_costume_name');
      if(!nameWidget) {
        warn('new_costume_name widget not found');
      } else {
        const createCostumeCallback = async () => {
          const newName = (nameWidget?.value || '').trim();
          const character = charWidget.value;
          log('Create costume button clicked, newName:', newName, 'character:', character);
          if(!newName) {
            warn('Empty costume name');
            return;
          }
          if(!character || character === 'None') {
            warn('No character selected');
            return;
          }
          log('Creating costume', newName, 'for', character);
          try {
            log('Sending fetch to /vnccs/create_costume');
            const response = await fetch(`/vnccs/create_costume?character=${encodeURIComponent(character)}&costume=${encodeURIComponent(newName)}`);
            log('Response status:', response.status);
            const result = await response.json();
            log('Response result:', result);
            if(result.ok) {
              log('Costume created successfully:', newName);
              const cfg = await fetchConfig(character);
              log('Fetched config for update');
              updateCostumeList(node, cfg);
              costumeWidget.value = newName;
              log('Selected new costume:', newName);
              nameWidget.value = "";
              log('Cleared new_costume_name field');
            } else {
              warn('Create failed:', result.error);
            }
          } catch(e) {
            console.error('Error creating costume:', e);
            warn('Error creating costume:', e);
          }
        };
        const button = node.addWidget('button', 'Create New Costume', '', createCostumeCallback);
        if(button) {
          button._vnccsCreateCostume = true;
          log('Create costume button added');
        }
      }
    }

    let updatingCharacter = false;
    let updatingCostume = false;

    const origChar = charWidget.callback;
    charWidget.callback = async function(){
      log('character callback triggered, value:', charWidget.value, 'updatingCharacter:', updatingCharacter);
      if(updatingCharacter) {
        log('already updating character, skipping');
        return;
      }
      updatingCharacter = true;
      log('processing character change');
      if(origChar) try{ origChar.apply(this, arguments); } catch(e){ warn('orig char callback error', e); }
      log('fetching config for character:', charWidget.value);
      try{
        const cfg = await fetchConfig(charWidget.value);
        log('config fetched, updating costume list');
        updateCostumeList(node, cfg);
      } catch(e){ warn('update costume list error', e); }
      updatingCharacter = false;
      log('character update completed');
    };

    const origCostume = costumeWidget.callback;
    costumeWidget.callback = async function(){
      log('costume callback triggered, value:', costumeWidget.value, 'updatingCostume:', updatingCostume);
      if(updatingCostume) {
        log('already updating costume, skipping');
        return;
      }
      updatingCostume = true;
      log('processing costume change');
      if(origCostume) try{ origCostume.apply(this, arguments); } catch(e){ warn('orig costume callback error', e); }
      log('fetching config for costume update, character:', charWidget.value);
      try{
        const cfg = await fetchConfig(charWidget.value);
        log('config fetched, applying costume data');
        applyCostumeData(node, cfg, costumeWidget.value);
      } catch(e){ warn('apply costume data error', e); }
      updatingCostume = false;
      log('costume update completed');
    };
    log('callbacks set for CharacterAssetSelector');
    
    setTimeout(async ()=>{
      log('Initial load for CharacterAssetSelector', node.id);
      const charWidget = node.widgets?.find(w => w.name === 'character');
      if (charWidget && charWidget.value && charWidget.value !== 'None') {
        log('Loading initial config for character:', charWidget.value);
        try {
          const cfg = await fetchConfig(charWidget.value);
          log('Initial config loaded, updating costume list');
          updateCostumeList(node, cfg);
          log('Initial load completed');
        } catch(e) {
          warn('Initial load error:', e);
        }
      } else {
        log('No character selected for initial load');
      }
    },300);
  }
  function hook(node){
    if(!node) return;
    const title = (node.title || '').trim();
    if(node.comfyClass === 'CharacterCreator' || title === 'VNCCS Character Creator'){
      if(node.__vnccsHooked) return;
      node.__vnccsHooked = true;
      log('hook Creator', node.id);
      const sel=(node.widgets||[]).find(w=>w.name==='existing_character'); if(!sel){ warn('no existing_character'); return; }
  addCreateButtonSafely(node);
    const orig=sel.callback; sel.callback=async function(){ 
      if(orig) try{ orig.apply(this,arguments);}catch(e){} 
      let characterName = sel.value;
      log('existing_character callback triggered, sel.value:', characterName, 'type:', typeof characterName);
      
      if (typeof characterName !== 'string') {
        if (characterName && characterName.title) {
          characterName = characterName.title;
          log('Extracted name from node title:', characterName);
        } else if (characterName && characterName.id !== undefined) {
          characterName = 'node_' + characterName.id;
          log('Extracted name from node id:', characterName);
        } else {
          log('Cannot extract string name from sel.value, using fallback');
          characterName = 'unknown';
        }
      }
      
      const cfg=await fetchConfig(characterName); 
      apply(node,cfg); 
    };
    if(!node.widgets.find(w=>w.name==='vnccs_reload')){
      node.addWidget('button','Reload Config','', async ()=>{ 
        let characterName = sel.value;
        log('Reload Config button clicked, sel.value:', characterName, 'type:', typeof characterName);
        
        if (typeof characterName !== 'string') {
          if (characterName && characterName.title) {
            characterName = characterName.title;
          } else if (characterName && characterName.id !== undefined) {
            characterName = 'node_' + characterName.id;
          } else {
            characterName = 'unknown';
          }
        }
        
        const cfg=await fetchConfig(characterName); 
        apply(node,cfg); 
      });
    }
    setTimeout(async ()=>{
      let characterName = sel.value;
      log('Initial load, sel.value:', characterName, 'type:', typeof characterName);
      
      if (typeof characterName !== 'string') {
        if (characterName && characterName.title) {
          characterName = characterName.title;
        } else if (characterName && characterName.id !== undefined) {
          characterName = 'node_' + characterName.id;
        } else {
          characterName = 'unknown';
        }
      }
      
      const cfg=await fetchConfig(characterName); 
      apply(node,cfg); 
    },200);
    } else if(
      (node.comfyClass && node.comfyClass.startsWith('CharacterAssetSelector')) ||
      (title && title.toLowerCase().startsWith('vnccs character selector'))
    ){
      hookCharacterAssetSelector(node);
    }
  }
  function scan(){ try{ (app.graph?._nodes||[]).forEach(hook); }catch(e){} }
  function register(){
    if(!window.app || !app.graph){ return setTimeout(register,150); }
    if(app.registerExtension){
      app.registerExtension({ name:'vnccs.autofill', nodeCreated(node){ hook(node); }, loadedGraph(){ scan(); } });
      log('extension registered');
    } else {
      // Fallback periodic scan
      setInterval(scan, 500);
      log('fallback scan mode');
    }
    scan();
  }
  register();
})();
