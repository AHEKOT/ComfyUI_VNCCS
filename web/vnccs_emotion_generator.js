import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "VNCCS.EmotionGenerator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EmotionGenerator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                this.addWidget("button", "Add Emotion", null, () => {
                    const emotionsWidget = this.widgets.find(w => w.name === "emotions");
                    const selectorWidget = this.widgets.find(w => w.name === "emotion_selector");
                    
                    if (emotionsWidget && selectorWidget) {
                        const currentEmotions = emotionsWidget.value || "";
                        const selectedEmotion = selectorWidget.value;
                        
                        const emotionsList = currentEmotions.split(/[,\n]/).map(e => e.trim()).filter(e => e);
                        
                        if (emotionsList.includes(selectedEmotion)) {
                            return;
                        }
                        
                        let newEmotions = currentEmotions;
                        if (newEmotions && !newEmotions.endsWith(",")) {
                            newEmotions += ",";
                        }
                        newEmotions += selectedEmotion;
                        
                        emotionsWidget.value = newEmotions;
                        
                        selectorWidget.value = selectorWidget.options.values[0];
                        
                        if (emotionsWidget.callback) {
                            emotionsWidget.callback(emotionsWidget.value);
                        }
                    }
                });
                
                return result;
            };
        }
    }
});