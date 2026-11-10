"""Three-class presentation; keeps the retained four-class preview intact."""
from pathlib import Path
from PIL import ImageFont
from .flower_overlay import FlowerOverlays,FlowerIntroduction
from .markdown_panel import MarkdownPanel
INTRODUCTIONS={
 0:'## 芍药 · 花开有约\n层叠花瓣与柔和粉色，让桌面多一份温柔。\n\n> 欣赏花开，感受日常美好。',
 1:'## 百合 · 雅致相伴\n舒展的花姿与明亮的色彩，为空间增添灵动。\n\n> 一束美好，点亮心情。',
 2:'## 菊花 · 明媚绽放\n层叠的花瓣与温暖的黄色，带来充满活力的观赏效果。\n\n> 让生活多一点阳光。',
}
class ThreeFlowerIntroduction(FlowerIntroduction):
 def __init__(self,class_id):
  super().__init__(class_id);self.panel=MarkdownPanel(INTRODUCTIONS[class_id]);root=Path(__file__).with_name('fonts')
  self.panel.font=ImageFont.truetype(str(root/'NotoSansCJKsc-ThreeFlowers.otf'),14);self.panel.bold_font=ImageFont.truetype(str(root/'NotoSansCJKsc-ThreeFlowers-Bold.otf'),14);self.panel.heading_font=ImageFont.truetype(str(root/'NotoSansCJKsc-ThreeFlowers-Bold.otf'),18);self.font=self.panel.font
class ThreeFlowerOverlays(FlowerOverlays):
 def update(self,detections,now):
  kept=super().update([d for d in detections if d['class']<3],now)
  for t in self.tracks.values():
   if not isinstance(t['overlay'],ThreeFlowerIntroduction):
    old=t['overlay'];new=ThreeFlowerIntroduction(t['class']);new.__dict__.update({k:v for k,v in old.__dict__.items() if k not in ['panel','font','_cached_text','_cached_banner']});t['overlay']=new
  return kept
