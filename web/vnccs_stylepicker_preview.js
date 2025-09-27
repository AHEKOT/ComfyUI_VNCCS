// Веб-интерфейс для предварительного просмотра стилей в VNCCS Style Picker

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "VNCCS.StylePickerPreview",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VNCCSStylePicker") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.addWidget("image_preview", "VNCCS_PREVIEW", "", {
                    serialize: false,
                    hideOnZoom: false
                });
                
                this.stylePreviewImg = null;
                
                this.updateNodeSize = function() {
                    let requiredHeight = 50;
                    if (this.widgets) {
                        for (let i = 0; i < this.widgets.length; i++) {
                            const widget = this.widgets[i];
                            if (widget.name !== "VNCCS_PREVIEW") {
                                requiredHeight += 30;
                            }
                        }
                    }
                    requiredHeight += 150;
                    
                    const minWidth = 300;
                    const minHeight = requiredHeight;
                    
                    this.size = [Math.max(this.size[0], minWidth), Math.max(this.size[1], minHeight)];
                };
                
                this.updateNodeSize();
                
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    if (originalOnResize) {
                        originalOnResize.apply(this, arguments);
                    }
                    this.setDirtyCanvas(true, true);
                };
                
                setTimeout(() => {
                    this.startStyleWatcher();
                }, 200);
                
                return r;
            };
            
            const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (originalOnDrawForeground) {
                    originalOnDrawForeground.apply(this, arguments);
                }
                
                this.drawStylePreview(ctx);
            };
            
            nodeType.prototype.drawStylePreview = function(ctx) {
                let widgetsHeight = 50;
                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const widget = this.widgets[i];
                        if (widget.name !== "VNCCS_PREVIEW") {
                            widgetsHeight += 30;
                        }
                    }
                }
                widgetsHeight += 10;
                
                const previewX = 10;
                const previewY = widgetsHeight;
                const previewW = this.size[0] - 20;
                const availableHeight = this.size[1] - widgetsHeight - 20;
                const previewH = Math.max(120, Math.min(availableHeight, 300));
                
                if (!this.stylePreviewImg || !this.stylePreviewImg.complete || this.stylePreviewImg.naturalWidth === 0) {
                    ctx.fillStyle = "#333";
                    ctx.fillRect(previewX, previewY, previewW, previewH);
                    
                    ctx.strokeStyle = "#666";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(previewX, previewY, previewW, previewH);
                    
                    ctx.fillStyle = "#999";
                    ctx.font = "14px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Preview Loading...", previewX + previewW/2, previewY + previewH/2);
                } else {
                    const availableHeight = this.size[1] - widgetsHeight - 20;
                    const previewH = Math.max(120, Math.min(availableHeight, 300));
                    
                    const imgAspect = this.stylePreviewImg.naturalWidth / this.stylePreviewImg.naturalHeight;
                    const previewAspect = previewW / previewH;
                    
                    let drawW, drawH, drawX, drawY;
                    if (imgAspect > previewAspect) {
                        drawW = previewW;
                        drawH = previewW / imgAspect;
                        drawX = previewX;
                        drawY = previewY + (previewH - drawH) / 2;
                    } else {
                        drawW = previewH * imgAspect;
                        drawH = previewH;
                        drawX = previewX + (previewW - drawW) / 2;
                        drawY = previewY;
                    }
                    
                    ctx.fillStyle = "#222";
                    ctx.fillRect(previewX, previewY, previewW, previewH);
                    
                    ctx.drawImage(this.stylePreviewImg, drawX, drawY, drawW, drawH);
                    
                    ctx.strokeStyle = "#666";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(previewX, previewY, previewW, previewH);
                }
            };
            
            nodeType.prototype.updateStylePreview = function() {
                if (!this.widgets) return;
                
                const artistWidget = this.widgets.find(w => w.name === "artist_style");
                const studioWidget = this.widgets.find(w => w.name === "studio_style");
                
                if (!artistWidget || !studioWidget) return;
                
                const selectedArtist = artistWidget.value;
                const selectedStudio = studioWidget.value;
                
                let selectedStyle;
                
                const artistChanged = selectedArtist !== this._lastArtist;
                const studioChanged = selectedStudio !== this._lastStudio;
                
                this._lastArtist = selectedArtist;
                this._lastStudio = selectedStudio;
                
                if (artistChanged && selectedArtist !== "None") {
                    selectedStyle = selectedArtist;
                } else if (studioChanged && selectedStudio !== "None") {
                    selectedStyle = selectedStudio;
                } else if (selectedArtist !== "None") {
                    selectedStyle = selectedArtist;
                } else if (selectedStudio !== "None") {
                    selectedStyle = selectedStudio;
                } else {
                    selectedStyle = "None";
                }
                
                
                let imagePath;
                if (selectedStyle === "None" || !selectedStyle) {
                    this.stylePreviewImg = null;
                    this.setDirtyCanvas(true, true);
                    return;
                } else {
                    const cleanName = selectedStyle
                        .replace(/[<>:"/\\|?*]/g, '_')
                        .replace(/\s+/g, '_')
                        .replace(/[()]/g, '');
                    imagePath = `/extensions/VNCCS/styles/${cleanName}.jpg`;
                }
                
                const img = new Image();
                img.onload = () => {
                    this.stylePreviewImg = img;
                    this.updateNodeSize();
                    this.setDirtyCanvas(true, true);
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                };
                img.onerror = () => {
                    this.stylePreviewImg = null;
                    this.setDirtyCanvas(true, true);
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                };
                img.src = imagePath;
            };
            
            const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(widget, value, oldValue) {
                const r = origOnWidgetChanged ? origOnWidgetChanged.apply(this, arguments) : undefined;
                
                if ((widget.name === "artist_style" || widget.name === "studio_style") && value !== oldValue) {
                    setTimeout(() => {
                        this.updateStylePreview();
                    }, 10);
                }
                
                return r;
            };
            
            const origSetWidgetValue = nodeType.prototype.setWidgetValue;
            nodeType.prototype.setWidgetValue = function(widget, value, force) {
                const oldValue = widget.value;
                const r = origSetWidgetValue ? origSetWidgetValue.apply(this, arguments) : undefined;
                
                if ((widget.name === "artist_style" || widget.name === "studio_style") && value !== oldValue) {
                    setTimeout(() => {
                        this.updateStylePreview();
                    }, 10);
                }
                
                return r;
            };
            
            nodeType.prototype.startStyleWatcher = function() {
                if (this.styleWatcherInterval) {
                    clearInterval(this.styleWatcherInterval);
                }
                
                let lastArtistValue = null;
                let lastStudioValue = null;
                
                this.styleWatcherInterval = setInterval(() => {
                    if (!this.widgets) return;
                    
                    const artistWidget = this.widgets.find(w => w.name === "artist_style");
                    const studioWidget = this.widgets.find(w => w.name === "studio_style");
                    
                    if (!artistWidget || !studioWidget) return;
                    
                    const currentArtist = artistWidget.value;
                    const currentStudio = studioWidget.value;
                    
                    const artistChanged = currentArtist !== lastArtistValue;
                    const studioChanged = currentStudio !== lastStudioValue;
                    
                    if (artistChanged || studioChanged) {
                        lastArtistValue = currentArtist;
                        lastStudioValue = currentStudio;
                        this.updateStylePreview();
                    }
                }, 100);
            };
            
            nodeType.prototype.stopStyleWatcher = function() {
                if (this.styleWatcherInterval) {
                    clearInterval(this.styleWatcherInterval);
                    this.styleWatcherInterval = null;
                }
            };
            
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                setTimeout(() => {
                    this.updateStylePreview();
                    this.startStyleWatcher();
                }, 100);
                return r;
            };
            
            const origOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                this.stopStyleWatcher();
                if (origOnRemoved) {
                    origOnRemoved.apply(this, arguments);
                }
            };
            
            const origOnMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function(event, pos, canvasNode) {
                const r = origOnMouseDown ? origOnMouseDown.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    this.updateStylePreview();
                }, 50);
                
                return r;
            };
        }
    }
});
