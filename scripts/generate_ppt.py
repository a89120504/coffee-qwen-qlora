# -*- coding: utf-8 -*-
"""generate_ppt.py - 生成期末簡報 PPT"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path

# ========== 配色方案 ==========
BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)
BG_CARD = RGBColor(0x16, 0x21, 0x3E)
ACCENT_GOLD = RGBColor(0xD4, 0xA5, 0x74)
ACCENT_BLUE = RGBColor(0x00, 0xD2, 0xFF)
ACCENT_GREEN = RGBColor(0x00, 0xE6, 0x76)
ACCENT_RED = RGBColor(0xFF, 0x6B, 0x6B)
TEXT_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEXT_LIGHT = RGBColor(0xCC, 0xCC, 0xCC)
TEXT_DIM = RGBColor(0x99, 0x99, 0x99)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=TEXT_WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Microsoft JhengHei"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_paragraph(text_frame, text, font_size=16, color=TEXT_WHITE, bold=False,
                  alignment=PP_ALIGN.LEFT, space_before=Pt(6), font_name="Microsoft JhengHei"):
    p = text_frame.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    if space_before:
        p.space_before = space_before
    return p


def add_table(slide, left, top, width, height, rows, cols, data, header_color=ACCENT_GOLD):
    table_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table
    for i, row_data in enumerate(data):
        for j, cell_text in enumerate(row_data):
            cell = table.cell(i, j)
            cell.text = str(cell_text)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                paragraph.font.name = "Microsoft JhengHei"
                if i == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
                else:
                    paragraph.font.color.rgb = TEXT_WHITE
            # 背景色
            fill = cell.fill
            fill.solid()
            if i == 0:
                fill.fore_color.rgb = header_color
            else:
                fill.fore_color.rgb = RGBColor(0x22, 0x2E, 0x50) if i % 2 == 1 else RGBColor(0x1A, 0x25, 0x42)
    return table


def make_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, BG_DARK)
    # 咖啡 emoji + 標題
    add_text_box(slide, 0.5, 1.0, 9, 1.2, "☕ Coffee Sommelier", 44, ACCENT_GOLD, True, PP_ALIGN.CENTER)
    add_text_box(slide, 0.5, 2.2, 9, 0.8, "基於 QLoRA 微調的咖啡品鑑助手", 28, TEXT_WHITE, False, PP_ALIGN.CENTER)
    add_text_box(slide, 0.5, 3.5, 9, 0.5, "Qwen2.5-3B-Instruct × Coffee Reviews Dataset", 18, ACCENT_BLUE, False, PP_ALIGN.CENTER)
    add_text_box(slide, 0.5, 5.5, 9, 0.5, "LLM 期末專案報告  |  2026.06.12", 16, TEXT_DIM, False, PP_ALIGN.CENTER)


def make_motivation_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.3, 9, 0.6, "🎯 動機與目標", 32, ACCENT_GOLD, True)

    txBox = add_text_box(slide, 0.5, 1.2, 9, 1.5, "❌ 問題：通用 LLM 回答咖啡問題時...", 20, ACCENT_RED, True)
    tf = txBox.text_frame
    add_paragraph(tf, "  • 編造不存在的品牌名稱", 16, TEXT_LIGHT)
    add_paragraph(tf, "  • 缺乏具體的杯測評分與產區資訊", 16, TEXT_LIGHT)
    add_paragraph(tf, "  • 沖煮建議泛泛而談，沒有精確參數", 16, TEXT_LIGHT)

    txBox2 = add_text_box(slide, 0.5, 3.5, 9, 1.5, "✅ 目標：打造專業咖啡品鑑師 AI", 20, ACCENT_GREEN, True)
    tf2 = txBox2.text_frame
    add_paragraph(tf2, "  • 推薦真實存在的咖啡豆（含烘焙坊、產地）", 16, TEXT_LIGHT)
    add_paragraph(tf2, "  • 引用真實杯測評分（CoffeeReview.com 數據）", 16, TEXT_LIGHT)
    add_paragraph(tf2, "  • 提供精確沖煮參數（水溫、粉水比、研磨度）", 16, TEXT_LIGHT)


def make_dataset_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.3, 9, 0.6, "📊 Model & Dataset", 32, ACCENT_GOLD, True)

    # 模型資訊
    add_text_box(slide, 0.5, 1.1, 4, 0.4, "模型選擇", 20, ACCENT_BLUE, True)
    data1 = [
        ["項目", "內容"],
        ["模型", "Qwen2.5-3B-Instruct"],
        ["參數量", "31 億"],
        ["選擇理由", "中文能力強、適合 8GB VRAM"],
    ]
    add_table(slide, 0.5, 1.6, 4, 1.5, 4, 2, data1)

    # 資料集資訊
    add_text_box(slide, 5.2, 1.1, 4.5, 0.4, "資料集", 20, ACCENT_BLUE, True)
    data2 = [
        ["項目", "內容"],
        ["來源", "Kaggle Coffee Reviews"],
        ["原始筆數", "2,095 筆咖啡評鑑"],
        ["訓練資料", "8,386 筆中文問答對"],
    ]
    add_table(slide, 5.2, 1.6, 4.5, 1.5, 4, 2, data2)

    # 資料擴增
    add_text_box(slide, 0.5, 3.8, 9, 0.4, "資料處理流程", 20, ACCENT_BLUE, True)
    txBox = add_text_box(slide, 0.5, 4.3, 9, 2, "英文 CSV → 風味關鍵字翻譯 → 4 類中文問答對生成 → ChatML 格式", 16, TEXT_LIGHT)
    tf = txBox.text_frame
    add_paragraph(tf, "① 單品咖啡風味描述  ② 依偏好推薦咖啡  ③ 產地特色介紹  ④ 沖煮建議", 14, TEXT_DIM)


def make_finetuning_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.3, 9, 0.6, "⚙️ Fine-Tuning Hyperparameters", 32, ACCENT_GOLD, True)

    data = [
        ["超參數", "值", "說明"],
        ["量化方式", "NF4 (4-bit)", "QLoRA 核心"],
        ["LoRA Rank", "16", "低秩分解維度"],
        ["LoRA Alpha", "32", "縮放因子"],
        ["Learning Rate", "2e-4", "cosine scheduler"],
        ["Batch Size", "2 × 4 = 有效 8", "gradient accumulation"],
        ["Epochs", "3", ""],
        ["Max Seq Length", "1024", ""],
        ["Optimizer", "paged_adamw_8bit", "節省 VRAM"],
        ["可訓練參數", "0.96%", "29.9M / 3.1B"],
    ]
    add_table(slide, 0.5, 1.1, 9, 3.5, 10, 3, data)

    # 硬體限制
    add_text_box(slide, 0.5, 5.0, 9, 0.4, "💡 硬體限制解決方案", 18, ACCENT_BLUE, True)
    add_text_box(slide, 0.5, 5.4, 9, 0.8,
                 "RTX 3070 (8GB) → QLoRA 4-bit 量化 + Gradient Checkpointing + Paged Optimizer → 實際用量 ~6GB",
                 14, TEXT_LIGHT)


def make_prompt_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.2, 9, 0.5, "💬 Prompt Engineering", 32, ACCENT_GOLD, True)

    # 左欄：System Prompt
    add_text_box(slide, 0.3, 0.9, 4.5, 0.35, "📋 System Prompt（角色設定）", 17, ACCENT_BLUE, True)
    sys_prompt = (
        "你是一位專業的咖啡品鑑師與顧問，\n"
        "擁有豐富的精品咖啡知識。\n"
        "你熟悉世界各產區的咖啡特色、\n"
        "不同烘焙程度的風味差異、\n"
        "以及各種沖煮方法。\n"
        "請用專業但易懂的方式，\n"
        "以繁體中文回答使用者關於\n"
        "咖啡的各種問題。"
    )
    add_text_box(slide, 0.3, 1.35, 4.5, 2.5, sys_prompt, 12, TEXT_LIGHT, font_name="Microsoft JhengHei")

    # 右欄：實際運作的 Prompt Template
    add_text_box(slide, 5.2, 0.9, 4.5, 0.35, "🔧 Prompt Template（ChatML）", 17, ACCENT_GREEN, True)
    template = (
        "<|im_start|>system\n"
        "{System Prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "{使用者問題}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{模型回答}\n"
        "<|im_end|>"
    )
    add_text_box(slide, 5.2, 1.35, 4.5, 2.5, template, 12, ACCENT_GREEN, font_name="Consolas")

    # 設計決策表格
    add_text_box(slide, 0.5, 4.2, 9, 0.35, "🎯 Prompt 設計決策", 18, ACCENT_BLUE, True)
    data = [
        ["設計決策", "選擇", "效果"],
        ["角色定位", "咖啡品鑑師", "回答語氣更專業"],
        ["語言指定", "繁體中文", "貼近台灣使用者"],
        ["回答結構", "推薦→風味→沖煮", "三段式，資訊結構化"],
        ["格式", "Qwen ChatML", "與模型原生格式對齊"],
    ]
    add_table(slide, 0.5, 4.65, 9, 2.0, 5, 3, data)


def make_training_result_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.3, 9, 0.6, "📈 訓練結果", 32, ACCENT_GOLD, True)

    data = [
        ["項目", "數值"],
        ["訓練時間", "4 小時 8 分鐘"],
        ["總步數", "726 steps"],
        ["起始 Loss", "~0.50"],
        ["最終 Loss", "0.12 ✅"],
        ["可訓練參數", "0.96% (29.9M / 3.1B)"],
        ["LoRA Adapter 大小", "~120 MB"],
        ["GPU 使用", "RTX 3070 8GB (實際 ~6GB)"],
    ]
    add_table(slide, 1.5, 1.2, 7, 3.5, 8, 2, data)

    add_text_box(slide, 0.5, 5.2, 9, 0.8,
                 "💡 Loss 從 0.50 穩定收斂至 0.12，顯示模型成功學習了咖啡領域知識",
                 16, ACCENT_GREEN, True, PP_ALIGN.CENTER)


def make_ab_slide_1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.2, 9, 0.5, "🔬 A/B 對比 — 推薦花香莓果咖啡", 28, ACCENT_GOLD, True)

    # Before
    add_text_box(slide, 0.3, 0.9, 4.5, 0.4, "🔵 微調前（原始模型）", 18, ACCENT_RED, True)
    txBox1 = add_text_box(slide, 0.3, 1.4, 4.5, 3.5,
                          "推薦「Colônia de São João」", 14, TEXT_LIGHT)
    tf1 = txBox1.text_frame
    add_paragraph(tf1, "❌ 品牌不存在（虛構）", 13, ACCENT_RED)
    add_paragraph(tf1, "❌ 無產地資訊", 13, ACCENT_RED)
    add_paragraph(tf1, "❌ 無杯測評分", 13, ACCENT_RED)
    add_paragraph(tf1, "❌ 沖煮建議模糊", 13, ACCENT_RED)
    add_paragraph(tf1, "", 8)
    add_paragraph(tf1, "「淺焙咖啡通常較容易溶解，", 12, TEXT_DIM)
    add_paragraph(tf1, "在溫暖的水中會釋放更多風味...」", 12, TEXT_DIM)

    # After
    add_text_box(slide, 5.2, 0.9, 4.5, 0.4, "🟢 微調後（咖啡品鑑師）", 18, ACCENT_GREEN, True)
    txBox2 = add_text_box(slide, 5.2, 1.4, 4.5, 3.5,
                          "推薦 Colombia Pink Bourbon Honey", 14, TEXT_LIGHT)
    tf2 = txBox2.text_frame
    add_paragraph(tf2, "✅ 烘焙商：Bird Rock Coffee Roasters", 13, ACCENT_GREEN)
    add_paragraph(tf2, "✅ 產地：Huila Department", 13, ACCENT_GREEN)
    add_paragraph(tf2, "✅ 品鑑評分：93/100", 13, ACCENT_GREEN)
    add_paragraph(tf2, "✅ 風味：茉莉花、莓果、蜂蜜", 13, ACCENT_GREEN)
    add_paragraph(tf2, "", 8)
    add_paragraph(tf2, "「手沖，水溫 90-93°C，", 12, TEXT_DIM)
    add_paragraph(tf2, "粉水比 1:15，中細研磨」", 12, TEXT_DIM)


def make_ab_slide_2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.2, 9, 0.5, "🔬 A/B 對比 — 整體評估", 28, ACCENT_GOLD, True)

    data = [
        ["評估維度", "🔵 微調前", "🟢 微調後"],
        ["推薦精確度", "❌ 虛構品牌", "✅ 真實豆名+烘焙坊"],
        ["專業數據", "❌ 無", "✅ 杯測評分+產區"],
        ["沖煮建議", "❌ 泛泛而談", "✅ 精確溫度/比例/研磨度"],
        ["回答風格", "冗長通用", "簡潔專業"],
        ["語言品質", "中英夾雜", "純繁體中文"],
        ["引用來源", "無可驗證資料", "CoffeeReview.com 真實數據"],
    ]
    add_table(slide, 0.5, 1.0, 9, 3.5, 7, 3, data)

    add_text_box(slide, 0.5, 5.0, 9, 0.8,
                 "💡 微調後模型學會了「用真實資料說話」，而非泛泛而談或編造資訊",
                 16, ACCENT_GREEN, True, PP_ALIGN.CENTER)


def make_conclusion_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 0.3, 9, 0.6, "🏁 結論與展望", 32, ACCENT_GOLD, True)

    txBox = add_text_box(slide, 0.5, 1.2, 9, 1.5, "結論", 22, ACCENT_BLUE, True)
    tf = txBox.text_frame
    add_paragraph(tf, "✅ 成功在消費級 GPU（RTX 3070 8GB）上完成 3B 模型微調", 16, TEXT_LIGHT)
    add_paragraph(tf, "✅ QLoRA 技術：只訓練 0.96% 參數即可顯著提升領域能力", 16, TEXT_LIGHT)
    add_paragraph(tf, "✅ 模型從「泛泛回答」進化為「引用真實數據的專業品鑑師」", 16, TEXT_LIGHT)

    txBox2 = add_text_box(slide, 0.5, 3.5, 9, 1.5, "未來展望", 22, ACCENT_BLUE, True)
    tf2 = txBox2.text_frame
    add_paragraph(tf2, "🔹 加入更多咖啡評鑑網站資料，擴充知識庫", 16, TEXT_LIGHT)
    add_paragraph(tf2, "🔹 導入 RAG 技術，即時檢索最新咖啡資訊", 16, TEXT_LIGHT)
    add_paragraph(tf2, "🔹 部署為 LINE Bot / Web App，供使用者直接使用", 16, TEXT_LIGHT)


def make_thanks_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_text_box(slide, 0.5, 2.0, 9, 1, "☕ Thank You!", 48, ACCENT_GOLD, True, PP_ALIGN.CENTER)
    add_text_box(slide, 0.5, 3.5, 9, 0.6, "Q & A", 28, TEXT_WHITE, False, PP_ALIGN.CENTER)


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    make_title_slide(prs)        # 1
    make_motivation_slide(prs)   # 2
    make_dataset_slide(prs)      # 3
    make_finetuning_slide(prs)   # 4
    make_prompt_slide(prs)       # 5
    make_training_result_slide(prs)  # 6
    make_ab_slide_1(prs)         # 7
    make_ab_slide_2(prs)         # 8
    make_conclusion_slide(prs)   # 9
    make_thanks_slide(prs)       # 10

    out_path = Path(__file__).parent.parent / "results" / "Coffee_Sommelier_Presentation.pptx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"[完成] 簡報已儲存至: {out_path}")


if __name__ == "__main__":
    main()
