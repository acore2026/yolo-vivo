"""Small video-native Markdown renderer (no browser or network resources)."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Block:
    kind: str
    runs: list


def inline_runs(text, style='normal'):
    pattern = r'(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_|\[[^\]]+\]\([^)]+\))'
    runs = []
    for part in re.split(pattern, text):
        if not part:
            continue
        if part.startswith('`') and part.endswith('`'):
            runs.append((part[1:-1], 'code'))
        elif part.startswith(('**', '__')) and part.endswith(('**', '__')):
            runs.append((part[2:-2], 'bold'))
        elif part.startswith(('*', '_')) and part.endswith(('*', '_')):
            runs.append((part[1:-1], 'italic'))
        elif re.fullmatch(r'\[[^\]]+\]\([^)]+\)', part):
            runs.append((part[1:part.index(']')], 'link'))
        else:
            runs.append((part.replace('\\*', '*').replace('\\_', '_'), style))
    return runs


def parse_markdown(source):
    blocks, fenced = [], False
    for line in source.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        if fenced:
            blocks.append(Block('code', [(line or ' ', 'code')]))
        elif not line.strip():
            blocks.append(Block('space', []))
        elif re.fullmatch(r'\s*(?:-{3,}|\*{3,}|_{3,})\s*', line):
            blocks.append(Block('rule', []))
        elif match := re.match(r'^\s{0,3}(#{1,6})\s+(.+)', line):
            blocks.append(Block('heading', inline_runs(match[2])))
        elif match := re.match(r'^\s*>\s?(.*)', line):
            blocks.append(Block('quote', inline_runs(match[1])))
        elif match := re.match(r'^\s*[-+*]\s+(.+)', line):
            blocks.append(Block('list', [('• ', 'accent')] + inline_runs(match[1])))
        elif match := re.match(r'^\s*(\d+)[.)]\s+(.+)', line):
            blocks.append(Block('list', [(match[1] + '. ', 'accent')] + inline_runs(match[2])))
        else:
            blocks.append(Block('paragraph', inline_runs(line)))
    return blocks


class MarkdownPanel:
    WIDTH = 220
    PADDING = 14

    def __init__(self, source):
        self.blocks = parse_markdown(source)
        self.plain_text = '\n'.join(''.join(text for text, _ in block.runs)
                                    for block in self.blocks if block.runs)
        path = Path(__file__).with_name('fonts') / 'NotoSansCJKsc-Arrival.otf'
        self.font = ImageFont.truetype(str(path), 14)
        bold_path = path.with_name('NotoSansCJKsc-Arrival-Bold.otf')
        self.bold_font = ImageFont.truetype(str(bold_path), 14)
        self.heading_font = ImageFont.truetype(str(bold_path), 18)

    def render(self, count):
        commands, y, consumed = [], 14, 0
        for block in self.blocks:
            if consumed >= count:
                break
            if block.kind in ('space', 'rule'):
                if block.kind == 'rule':
                    commands.append(('rule', y))
                y += 7
                continue
            font = self.heading_font if block.kind == 'heading' else self.font
            line_height = 27 if block.kind == 'heading' else 21
            indent = 10 if block.kind in ('quote', 'code') else 0
            x = self.PADDING + indent
            if block.kind == 'quote':
                commands.append(('quote', y))
            for text, style in block.runs:
                run_font = self.bold_font if style == 'bold' and block.kind != 'heading' else font
                for index, char in enumerate(text):
                    if consumed >= count:
                        break
                    width = run_font.getlength(char)
                    # Keep closing Chinese punctuation with the preceding glyph.
                    following = text[index + 1:index + 2]
                    reserve = run_font.getlength(following) if following in '，。！？；：、）》」』' and following else 0
                    if x + width + reserve > self.WIDTH - self.PADDING:
                        x = self.PADDING + indent
                        y += line_height
                    commands.append(('char', x, y, char, style, run_font))
                    x += width
                    consumed += 1
            y += line_height + (4 if block.kind == 'heading' else 2)
            # The typing clock counts the plain-text block separator too.
            consumed += 1
        height = max(52, y + 10)
        yy, xx = np.mgrid[0:height, 0:self.WIDTH]
        mix = (xx / (self.WIDTH - 1) * .65 + yy / max(1, height - 1) * .35)[..., None]
        background = (np.array([12, 23, 43]) * (1 - mix) + np.array([38, 17, 55]) * mix).astype(np.uint8)
        canvas = Image.fromarray(background)
        draw = ImageDraw.Draw(canvas)
        # A restrained cyan-to-violet border with a softer inner highlight.
        border = Image.new('RGB', canvas.size)
        border_array = np.asarray(border).copy()
        ramp = np.linspace(0, 1, self.WIDTH)[None, :, None]
        border_array[:] = (np.array([64, 219, 244]) * (1 - ramp) + np.array([185, 103, 251]) * ramp).astype(np.uint8)
        outer = Image.new('L', canvas.size)
        ImageDraw.Draw(outer).rounded_rectangle((0, 0, self.WIDTH - 1, height - 1), radius=12, fill=200)
        ImageDraw.Draw(outer).rounded_rectangle((2, 2, self.WIDTH - 3, height - 3), radius=10, fill=0)
        canvas.paste(Image.fromarray(border_array), (0, 0), outer)
        draw = ImageDraw.Draw(canvas)
        colors = {'normal': '#DFE7F4', 'bold': '#FFFFFF', 'italic': '#CAB4FA',
                  'code': '#8FE5F2', 'link': '#75CFF5', 'accent': '#A9A0FC'}
        for command in commands:
            if command[0] == 'rule':
                draw.line((14, command[1], self.WIDTH - 14, command[1]), fill='#59456F')
            elif command[0] == 'quote':
                draw.rounded_rectangle((14, command[1], 16, command[1] + 19), radius=1, fill='#A78BFA')
            else:
                _, x, top, char, style, font = command
                if style == 'code':
                    draw.rectangle((x - 1, top, x + font.getlength(char) + 1, top + 19), fill='#24334F')
                draw.text((x, top), char, font=font, anchor='la', fill=colors[style])
                if style in ('link', 'italic'):
                    draw.line((x, top + 18, x + font.getlength(char), top + 18), fill=colors[style])
        return np.asarray(canvas)[:, :, ::-1].copy()
