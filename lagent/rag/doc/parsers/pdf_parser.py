import pdfplumber
import numpy as np
import logging
import re
from PIL import Image
import pytesseract
from copy import deepcopy
import os
from functools import cmp_to_key


class PdfParser:
    def __init__(self):
        self.boxes = []
        self.page_images = []
        self.page_chars = []
        self.mean_height = []
        self.mean_width = []
        self.page_cum_height = [0]
        self.left_chars = []
        self.text_content = []
        self.image_content = []

    def extract_text(self, pages):
        for page in pages:
            self.text_content.append(page.extract_text())

    def extract_images(self, pages, output_dir="extracted_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, page in enumerate(pages):
            img = page.to_image()
            img_path = os.path.join(output_dir, f"page_{i + 1}.png")
            img.save(img_path)
            logging.info(f"Saved image for page {i + 1} to {img_path}")

    def sort(self, boxes):
        def compare(b1, b2):
            top_diff = abs(b1["top"] - b2["top"])
            threshold = self.mean_height[b1["page_number"] - 1] / 2
            if top_diff > threshold:
                return b1["top"] - b2["top"]
            else:
                return b1["x1"] - b2["x1"]

        boxes = sorted(boxes, key=cmp_to_key(compare))
        return boxes

    def ocr(self, pagenum, img, chars, zoomin=3):
        bxs = [
            {
                "x0": c["x0"] / zoomin, "x1": c["x1"] / zoomin,  # 左和右的x
                "top": c["top"] / zoomin, "text": c.get("text", ""),  # 顶和底的y
                "bottom": c["bottom"] / zoomin,
                "page_number": pagenum
            }
            for c in chars
        ]
        bxs = self.sort(bxs)
        for b in bxs:
            if not b["text"]:
                left, right, top, bottom = b["x0"] * zoomin, b["x1"] * zoomin, b["top"] * zoomin, b["bottom"] * zoomin
                region = img.crop((left, top, right, bottom))
                b["text"] = pytesseract.image_to_string(region, lang="chi_sim+eng").strip()
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[-1] == 0:
            self.mean_height[-1] = np.median([b["bottom"] - b["top"] for b in bxs])
        for b in bxs:
            self.boxes.append(b)

    def images(self, file_path, zoomin=3, page_start=0, page_end=299):
        try:
            self.pdf = pdfplumber.open(file_path)
            pages = self.pdf.pages[page_start:page_end]
            self.page_images = [p.to_image(resolution=72 * zoomin).original for p in pages]
            self.page_chars = [[{**c, 'top': c['top'], 'bottom': c['bottom']} for c in page.dedupe_chars().chars] for
                               page in pages]
            self.extract_text(pages)
            self.extract_images(pages)
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            return

        for i, img in enumerate(self.page_images):
            chars = self.page_chars[i]
            self.mean_height.append(np.median([c["height"] for c in chars]) if chars else 0)
            self.mean_width.append(np.median([c["width"] for c in chars]) if chars else 8)
            self.page_cum_height.append(img.size[1] / zoomin)

            self.ocr(i + 1, img, chars, zoomin)

        if len(self.page_cum_height) != len(self.page_images) + 1:
            logging.warning("Mismatch in page cumulative height calculation.")

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12)
        ]:
            if re.match(p, line):
                return j
        return

    def x_dis(self, a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]),
                   abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def y_dis(self, a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def tag(self, box, zoomin):
        pn = [box["page_number"]]
        top = box["top"] - self.page_cum_height[pn[0] - 1]
        btm = box["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return " "
        while btm * zoomin > self.page_images[pn[-1] - 1].size[1]:
            btm -= self.page_images[pn[-1] - 1].size[1] / zoomin
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return " "
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##" \
            .format("-".join([str(p) for p in pn]), box["x0"], box["x1"], top, btm)

    def merge(self):
        def text_ends(box, suffix):
            return box.get("text", "").strip().endswith(suffix.strip())

        def text_starts(box, prefixes):
            text = box.get("text", "").strip()
            return any(text.startswith(prefix.strip()) for prefix in prefixes)

        bxs = self.boxes
        i = 0
        while i < len(bxs) - 1:
            current = bxs[i]
            next = bxs[i + 1]
            if current.get("layout_type", "") in ["table", "figure", "equation"]:
                i += 1
                continue

            vertical_dist = abs(self.y_dis(current, next))
            horizontal_dist = abs(self.x_dis(current, next))
            vertical_threshold = self.mean_height[current["page_number"] - 1] / 5
            horizontal_threshold = (current["x1"] - current["x0"] + next["x1"] - next["x0"]) / 2
            if text_ends(current, "， ") or text_starts(next, ["（", "）"]):
                horizontal_threshold = 8

            if vertical_dist < vertical_threshold and horizontal_dist <= horizontal_threshold:
                current["x1"] = next["x1"]
                current["top"] = (next["top"] + current["top"]) / 2
                current["bottom"] = (next["bottom"] + current["bottom"]) / 2
                current["text"] += next["text"]
                bxs.pop(i + 1)
                continue

            i += 1

        i = 0
        while i < len(bxs) - 1:
            current = bxs[i]
            next = bxs[i + 1]
            vertical_dist = abs(self.y_dis(current, next))
            vertical_threshold = self.mean_height[current["page_number"] - 1]
            vertical_threshold_5 = self.mean_height[current["page_number"] - 1] / 5
            if vertical_dist < vertical_threshold_5:
                i += 1
                continue
            if current["x1"] >= self.mean_width[current["page_number"] - 1] and vertical_dist <= vertical_threshold:
                current["x1"] = max(next["x1"], current["x1"])
                current["bottom"] = next["bottom"]
                current["text"] += next["text"]
                bxs.pop(i + 1)
                continue
            i += 1

        self.boxes = bxs

    def parse(self, file_path, zoomin=3):
        self.images(file_path, zoomin)
        self.merge()
        parsed_text = []
        for b in self.boxes:
            b["text"] += "\n"
            parsed_text.append(b["text"])
        self.text_content.append(parsed_text)
        for page_num, img in enumerate(self.page_images, start=1):
            image_path = f"image_page_{page_num}.png"
            img.save(image_path)  # 保存图像为文件
            self.image_content.append({
                "page_number": page_num,
                "image_path": image_path,
                "resolution": img.size
            })
        return {
            "text": self.text_content,
            "images": self.image_content
        }


def print_parsed_content(parsed_content):
    # 打印文本内容
    print("\nText:")
    if parsed_content.get("text"):
        for page_num, text in enumerate(parsed_content["text"], 1):
            print(f"\nPage {page_num}:\n{text}")
    else:
        print("No text")
    print("\nImages:")
    if "images" in parsed_content and parsed_content["images"]:
        for img_num, img in enumerate(parsed_content["images"], 1):
            print(f"\nImage {img_num}:")
            for key, value in img.items():
                print(f"{key}: {value}")
    else:
        print("No images")


def main():
    file_path = 'example.pdf'

    pdfparser = PdfParser()
    parsed_content = pdfparser.parse(file_path)

    print_parsed_content(parsed_content)


if __name__ == "__main__":
    main()
