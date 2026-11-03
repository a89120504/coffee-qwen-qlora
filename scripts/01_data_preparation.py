# -*- coding: utf-8 -*-
"""
01_data_preparation.py
將 Kaggle Coffee Reviews Dataset 轉換為中文 ChatML 微調格式
"""

import pandas as pd
import json
import random
import re
import os
from pathlib import Path

# ========== 翻譯字典 ==========

ROAST_DICT = {
    "Light": "淺焙", "Medium-Light": "中淺焙", "Medium": "中焙",
    "Medium-Dark": "中深焙", "Dark": "深焙",
}

ORIGIN_DICT = {
    "Ethiopia": "衣索比亞", "Colombia": "哥倫比亞", "Brazil": "巴西",
    "Kenya": "肯亞", "Guatemala": "瓜地馬拉", "Costa Rica": "哥斯大黎加",
    "Indonesia": "印尼", "India": "印度", "Mexico": "墨西哥",
    "Peru": "祕魯", "Honduras": "宏都拉斯", "El Salvador": "薩爾瓦多",
    "Nicaragua": "尼加拉瓜", "Panama": "巴拿馬", "Rwanda": "盧安達",
    "Burundi": "蒲隆地", "Tanzania": "坦尚尼亞", "Uganda": "烏干達",
    "Yemen": "葉門", "Jamaica": "牙買加", "Hawaii": "夏威夷",
    "Papua New Guinea": "巴布亞紐幾內亞", "Vietnam": "越南",
    "China": "中國", "Taiwan": "台灣", "Thailand": "泰國",
    "Myanmar": "緬甸", "Sumatra": "蘇門答臘", "Java": "爪哇",
    "Sulawesi": "蘇拉維西", "Bolivia": "玻利維亞",
    "Dominican Republic": "多明尼加", "Ecuador": "厄瓜多",
    "United States": "美國", "Blend": "混合產區",
}

FLAVOR_DICT = {
    "dark chocolate": "黑巧克力", "milk chocolate": "牛奶巧克力",
    "chocolate": "巧克力", "cocoa": "可可", "cacao": "可可",
    "jasmine": "茉莉花", "rose": "玫瑰", "lavender": "薰衣草",
    "honeysuckle": "金銀花", "floral": "花香", "flower": "花香",
    "blueberry": "藍莓", "strawberry": "草莓", "raspberry": "覆盆子",
    "blackberry": "黑莓", "cherry": "櫻桃", "berry": "莓果",
    "fruity": "果香", "fruit": "水果",
    "lemon": "檸檬", "orange": "柳橙", "grapefruit": "葡萄柚",
    "tangerine": "橘子", "citrus": "柑橘", "lime": "萊姆",
    "mango": "芒果", "papaya": "木瓜", "pineapple": "鳳梨",
    "passion fruit": "百香果", "tropical": "熱帶水果",
    "apple": "蘋果", "pear": "梨", "peach": "水蜜桃",
    "apricot": "杏桃", "plum": "李子", "grape": "葡萄",
    "almond": "杏仁", "hazelnut": "榛果", "walnut": "核桃",
    "peanut": "花生", "pecan": "胡桃", "nutty": "堅果",
    "caramel": "焦糖", "toffee": "太妃糖", "butterscotch": "奶油糖",
    "honey": "蜂蜜", "maple": "楓糖", "brown sugar": "黑糖",
    "molasses": "糖蜜", "syrup": "糖漿",
    "vanilla": "香草", "cinnamon": "肉桂", "clove": "丁香",
    "ginger": "薑", "cardamom": "荳蔻", "pepper": "胡椒",
    "smoky": "煙燻", "woody": "木質", "cedar": "雪松",
    "oak": "橡木", "earthy": "泥土", "herbal": "草本",
    "mint": "薄荷", "tea": "茶感",
    "sweet": "甜感", "balanced": "平衡", "complex": "複雜",
    "clean": "乾淨", "bright": "明亮", "crisp": "清脆",
    "smooth": "滑順", "creamy": "奶油感", "silky": "絲滑",
    "rich": "豐富", "full": "飽滿", "light": "輕盈",
    "tangy": "酸爽", "tart": "酸澀", "mellow": "醇和",
    "savory": "鹹香", "buttery": "奶油般", "velvety": "天鵝絨般",
    "juicy": "多汁", "winey": "酒香", "wine": "酒感",
    "rum": "蘭姆酒", "whiskey": "威士忌", "brandy": "白蘭地",
}

SYSTEM_PROMPT = (
    "你是一位專業的咖啡品鑑師與顧問，擁有豐富的精品咖啡知識。"
    "你熟悉世界各產區的咖啡特色、不同烘焙程度的風味差異、以及各種沖煮方法。"
    "請用專業但易懂的方式，以繁體中文回答使用者關於咖啡的各種問題，"
    "包括風味描述、咖啡推薦、沖煮建議等。"
)

BREW_TIPS = {
    "淺焙": "建議使用手沖方式，水溫 90-93°C，粉水比 1:15，中細研磨，能展現明亮的酸質與花果香氣。",
    "中淺焙": "建議使用手沖方式，水溫 89-92°C，粉水比 1:15，中細研磨，適合展現豐富的風味層次。",
    "中焙": "手沖或虹吸壺皆可，水溫 88-91°C，粉水比 1:14-1:15，中度研磨，口感均衡。",
    "中深焙": "適合義式濃縮或摩卡壺，水溫 86-90°C，粉水比 1:14，中粗研磨，醇厚度佳。",
    "深焙": "適合義式濃縮或法式濾壓壺，水溫 85-88°C，粉水比 1:13-1:14，粗研磨，口感濃郁厚實。",
}

ORIGIN_PROFILES = {
    "衣索比亞": {
        "style": "花香、柑橘、莓果、茶感與明亮酸質",
        "body": "口感通常較輕盈，層次感強",
        "best_for": "喜歡果香、花香、明亮酸質的人",
    },
    "哥倫比亞": {
        "style": "焦糖、堅果、巧克力、紅色水果與均衡甜感",
        "body": "酸甜平衡，口感圓潤，接受度高",
        "best_for": "想要穩定、均衡、不過度酸澀的人",
    },
    "肯亞": {
        "style": "黑醋栗、莓果、柑橘、番茄感與鮮明酸質",
        "body": "酸質清晰，風味集中，尾韻乾淨",
        "best_for": "喜歡高酸、果汁感和強烈風味的人",
    },
    "巴西": {
        "style": "堅果、可可、焦糖、低酸與厚實甜感",
        "body": "口感溫和厚實，常作為義式配方基底",
        "best_for": "不喜歡太酸、偏好巧克力堅果調的人",
    },
    "瓜地馬拉": {
        "style": "巧克力、香料、柑橘、蘋果酸與乾淨甜感",
        "body": "結構穩定，酸甜平衡，尾韻扎實",
        "best_for": "想要有層次但不過度尖銳酸質的人",
    },
    "哥斯大黎加": {
        "style": "蜂蜜、莓果、柑橘、花香與乾淨甜感",
        "body": "常見蜜處理，甜感突出且風味乾淨",
        "best_for": "喜歡甜感、果香和乾淨口感的人",
    },
    "巴拿馬": {
        "style": "花香、柑橘、熱帶水果、茶感與細緻甜感",
        "body": "風味細膩，常見高價精品批次",
        "best_for": "想嘗試高辨識度、細緻香氣的人",
    },
    "印尼": {
        "style": "草本、木質、香料、低酸與厚實口感",
        "body": "body 厚重，尾韻深沉",
        "best_for": "喜歡低酸、厚實、草本或香料調的人",
    },
}

ROAST_PROFILES = {
    "淺焙": {
        "flavor": "花香、果酸、茶感與產區特色最明顯",
        "risk": "萃取不足時容易尖酸、薄弱",
        "brew": "水溫 90-93°C，中細研磨，粉水比約 1:15，適合手沖或愛樂壓。",
    },
    "中淺焙": {
        "flavor": "兼具明亮酸質、甜感與較完整的風味層次",
        "risk": "研磨太粗會讓甜感不足，太細則可能苦澀",
        "brew": "水溫 89-92°C，中細研磨，粉水比約 1:15，適合手沖。",
    },
    "中焙": {
        "flavor": "酸甜較均衡，常見焦糖、堅果、可可與水果甜感",
        "risk": "過度萃取會讓苦味蓋過甜感",
        "brew": "水溫 88-91°C，中度研磨，粉水比 1:14-1:15，手沖、虹吸都合適。",
    },
    "中深焙": {
        "flavor": "巧克力、焦糖、堅果與較厚實 body",
        "risk": "水溫太高或研磨太細容易苦澀、焦味重",
        "brew": "水溫 86-90°C，中粗研磨，粉水比約 1:14，適合義式、摩卡壺或濾壓。",
    },
    "深焙": {
        "flavor": "低酸、厚實、煙燻、黑巧克力與烘焙風味明顯",
        "risk": "過度萃取會出現焦苦與乾澀",
        "brew": "水溫 85-88°C，粗研磨，粉水比 1:13-1:14，適合法式濾壓或義式濃縮。",
    },
}

PROCESS_PROFILES = {
    "日曬": {
        "method": "整顆咖啡果實連同果皮果肉一起乾燥，再去除外層。",
        "flavor": "常見濃郁果香、莓果、酒香、發酵甜感與較厚 body。",
        "risk": "若處理控制不好，可能有過度發酵或雜味。",
    },
    "水洗": {
        "method": "先去除果肉，再透過發酵與水洗去除果膠後乾燥。",
        "flavor": "風味乾淨、酸質清楚、層次分明，能凸顯產區特色。",
        "risk": "甜感和 body 可能不如日曬厚重。",
    },
    "蜜處理": {
        "method": "去除果皮後保留部分果膠乾燥，介於日曬與水洗之間。",
        "flavor": "常見蜂蜜甜感、果香、圓潤口感與乾淨尾韻。",
        "risk": "乾燥管理需要精準，否則風味容易不穩定。",
    },
    "厭氧發酵": {
        "method": "在低氧或密閉環境中控制發酵，之後再乾燥。",
        "flavor": "常見熱帶水果、酒香、香料或強烈發酵香氣。",
        "risk": "風味辨識度高，但有些人會覺得太濃或太像酒。",
    },
}


def generate_contextual_brew_tip(roast, flavors):
    """根據烘焙度與風味特色生成條件式沖煮建議，避免所有同烘焙度咖啡共用同一句"""
    base = ROAST_PROFILES.get(roast, ROAST_PROFILES["中焙"])
    parts = [base["brew"]]

    flower_flavors = {"花香", "茉莉花", "玫瑰", "薰衣草", "金銀花"}
    nut_choc_flavors = {"巧克力", "黑巧克力", "牛奶巧克力", "可可", "堅果", "杏仁", "榛果"}
    fruit_flavors = {"莓果", "藍莓", "草莓", "櫻桃", "柑橘", "檸檬", "柳橙", "果香"}
    ferment_flavors = {"酒香", "酒感", "蘭姆酒", "威士忌", "白蘭地"}

    flavor_set = set(flavors)
    if flavor_set & flower_flavors:
        parts.append("注水時建議放慢速度，讓花香層次更完整展開。")
    if flavor_set & nut_choc_flavors:
        parts.append("可稍微提高粉水比至 1:14，突出醇厚口感。")
    if flavor_set & fruit_flavors:
        parts.append("悶蒸時間可延長至 40 秒，讓果酸更圓潤。")
    if flavor_set & ferment_flavors:
        parts.append("建議水溫略降 1-2°C，避免放大酒精感。")

    return " ".join(parts)


def extract_flavors(text):
    """從英文描述中提取風味關鍵字並翻譯為中文"""
    if not isinstance(text, str) or not text.strip():
        return []
    text_lower = text.lower()
    found = []
    seen = set()
    # 先匹配長詞（避免 "dark chocolate" 被拆成 "dark" + "chocolate"）
    sorted_flavors = sorted(FLAVOR_DICT.items(), key=lambda x: len(x[0]), reverse=True)
    for en, cn in sorted_flavors:
        if en.lower() in text_lower and cn not in seen:
            found.append(cn)
            seen.add(cn)
    return found


def translate_origin(origin):
    """翻譯產地名稱"""
    if not isinstance(origin, str):
        return "未知產區"
    for en, cn in ORIGIN_DICT.items():
        if en.lower() in origin.lower():
            return cn
    return origin


def translate_roast(roast):
    """翻譯烘焙度"""
    if not isinstance(roast, str):
        return "中焙"
    return ROAST_DICT.get(roast.strip(), roast)


def make_conversation(user_msg, assistant_msg):
    """建立 ChatML 格式的對話"""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def generate_qa_pairs(row):
    """從一筆咖啡資料生成多組中文問答對"""
    pairs = []

    name = str(row.get("name", "")).strip()
    roaster = str(row.get("roaster", "")).strip()
    roast = translate_roast(row.get("roast", ""))
    origin = translate_origin(row.get("origin_1", ""))
    rating = row.get("rating", 0)

    # 合併描述欄位
    descs = []
    for col in ["desc_1", "desc_2", "desc_3"]:
        val = row.get(col, "")
        if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
            descs.append(val.strip())
    full_desc = " ".join(descs)

    if not full_desc or not name:
        return pairs

    # 提取風味
    flavors = extract_flavors(full_desc)
    flavor_str = "、".join(flavors[:6]) if flavors else "均衡醇和"
    brew_tip = generate_contextual_brew_tip(roast, flavors)

    # --- 類型 1：咖啡風味描述 ---
    q1_templates = [
        f"請介紹一下 {name} 這款咖啡的風味特色。",
        f"{roaster} 的 {name} 喝起來怎麼樣？",
        f"能描述一下 {name} 的風味嗎？",
        f"想了解 {name} 這支豆子，可以介紹一下嗎？",
        f"{name} 的風味和沖煮方式是什麼？",
    ]
    a1 = (
        f"**{name}**\n"
        f"- 烘焙商：{roaster}\n"
        f"- 烘焙度：{roast}\n"
        f"- 產地：{origin}\n"
        f"- 品鑑評分：{rating}/100\n\n"
        f"【風味特色】\n"
        f"這款咖啡的主要風味包括：{flavor_str}。\n\n"
        f"【沖煮建議】\n{brew_tip}"
    )
    q1 = random.choice(q1_templates)
    pairs.append(make_conversation(q1, a1))

    # --- 類型 2：基於偏好的推薦 ---
    if len(flavors) >= 2:
        pref_flavors = random.sample(flavors[:4], min(2, len(flavors[:4])))
        pref_str = "和".join(pref_flavors)
        q2_templates = [
            f"我喜歡帶有{pref_str}風味的咖啡，有推薦嗎？",
            f"想找{pref_str}調性的咖啡，有什麼好選擇？",
            f"有沒有{pref_str}風味比較突出的豆子？",
        ]
        q2 = random.choice(q2_templates)
        a2 = (
            f"根據你的偏好，我推薦 **{name}**（{roaster}）。\n\n"
            f"這款{roast}咖啡來自{origin}，品鑑評分 {rating}/100。"
            f"風味特色包括{flavor_str}，非常符合你對{pref_str}的喜好。\n\n"
            f"【沖煮建議】\n{brew_tip}"
        )
        pairs.append(make_conversation(q2, a2))

    # --- 類型 3：產地詢問 ---
    if origin != "未知產區" and origin != "混合產區":
        q3_templates = [
            f"{origin}的咖啡有什麼特色？",
            f"我想試試{origin}產區的咖啡，有推薦嗎？",
            f"聽說{origin}的咖啡不錯，可以推薦一款嗎？",
        ]
        q3 = random.choice(q3_templates)
        a3 = (
            f"{origin}是知名的咖啡產區。以 **{name}** 為例，"
            f"這款由 {roaster} 烘焙的{roast}咖啡，"
            f"展現出{flavor_str}等風味特色，品鑑評分達 {rating}/100。\n\n"
            f"【沖煮建議】\n{brew_tip}"
        )
        pairs.append(make_conversation(q3, a3))

    # --- 類型 4：烘焙度詢問 ---
    q4_templates = [
        f"推薦一款好喝的{roast}咖啡。",
        f"有沒有不錯的{roast}豆子可以推薦？",
        f"想喝{roast}的咖啡，有什麼建議？",
    ]
    q4 = random.choice(q4_templates)
    a4 = (
        f"推薦你試試 **{name}**（{roaster}），來自{origin}。\n\n"
        f"這款{roast}咖啡的風味特色是{flavor_str}，"
        f"品鑑評分 {rating}/100，品質出色。\n\n"
        f"【沖煮建議】\n{brew_tip}"
    )
    pairs.append(make_conversation(q4, a4))

    # --- 類型 5：複合條件推薦 ---
    if origin != "未知產區" and origin != "混合產區" and len(flavors) >= 2:
        combo_flavor = random.choice(flavors[:3])
        q5 = f"有沒有來自{origin}、帶{combo_flavor}風味的{roast}咖啡？"
        a5 = (
            f"符合你條件的推薦：**{name}**（{roaster}）。\n\n"
            f"- 產地：{origin} ✅\n"
            f"- 烘焙度：{roast} ✅\n"
            f"- 風味：{flavor_str}（包含{combo_flavor}） ✅\n"
            f"- 品鑑評分：{rating}/100\n\n"
            f"【沖煮建議】\n{brew_tip}"
        )
        pairs.append(make_conversation(q5, a5))

    # --- 類型 6：否定條件 ---
    negative_map = {
        "花香": "不要花香", "果香": "不要太多果酸",
        "巧克力": "不要巧克力調", "堅果": "不要堅果味",
    }
    for flav, neg_phrase in negative_map.items():
        if flav not in flavors and len(flavors) >= 2:
            q6 = f"推薦{neg_phrase}的咖啡。"
            a6 = (
                f"推薦 **{name}**（{roaster}），來自{origin}。\n\n"
                f"這款{roast}咖啡的風味以{flavor_str}為主，"
                f"不帶有明顯的{flav}調性，應該符合你的偏好。\n"
                f"品鑑評分 {rating}/100。\n\n"
                f"【沖煮建議】\n{brew_tip}"
            )
            pairs.append(make_conversation(q6, a6))
            break  # 每筆資料只生成一個否定條件

    return pairs


def generate_score_threshold_qa(df):
    """生成數值邊界正例/反例/邊界案例，教模型精確處理評分門檻"""
    pairs = []
    thresholds = [90, 92, 93, 95]

    for threshold in thresholds:
        above = df[df["rating"] >= threshold]
        below = df[(df["rating"] < threshold) & (df["rating"] >= threshold - 2)]

        # --- 正例：符合門檻 ---
        for _, row in above.sample(n=min(3, len(above)), random_state=42).iterrows():
            name = str(row.get("name", "")).strip()
            roaster = str(row.get("roaster", "")).strip()
            roast = translate_roast(row.get("roast", ""))
            origin = translate_origin(row.get("origin_1", ""))
            rating = row["rating"]
            descs = []
            for col in ["desc_1", "desc_2", "desc_3"]:
                val = row.get(col, "")
                if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
                    descs.append(val.strip())
            flavors = extract_flavors(" ".join(descs))
            flavor_str = "、".join(flavors[:6]) if flavors else "均衡醇和"
            brew_tip = generate_contextual_brew_tip(roast, flavors)

            q = f"推薦一款 {threshold} 分以上的咖啡。"
            a = (
                f"推薦 **{name}**（{roaster}），來自{origin}。\n\n"
                f"- 品鑑評分：{rating}/100 ✅（≥ {threshold} 分）\n"
                f"- 烘焙度：{roast}\n"
                f"- 風味：{flavor_str}\n\n"
                f"這款咖啡的評分達到 {rating} 分，符合你 {threshold} 分以上的要求。\n\n"
                f"【沖煮建議】\n{brew_tip}"
            )
            pairs.append(make_conversation(q, a))

        # --- 反例：不符合門檻，明確拒絕 ---
        for _, row in below.sample(n=min(2, len(below)), random_state=42).iterrows():
            name = str(row.get("name", "")).strip()
            rating = row["rating"]

            q = f"推薦一款 {threshold} 分以上的咖啡。"
            a = (
                f"**{name}** 的品鑑評分為 {rating}/100，"
                f"未達 {threshold} 分的門檻，因此不符合你的條件。\n\n"
                f"差距為 {threshold - rating:.0f} 分。"
                f"建議篩選評分確實達到 {threshold} 分以上的咖啡。"
            )
            pairs.append(make_conversation(q, a))

        # --- 邊界案例：剛好差 1 分 ---
        boundary = df[df["rating"] == threshold - 1]
        for _, row in boundary.sample(n=min(1, len(boundary)), random_state=42).iterrows():
            name = str(row.get("name", "")).strip()
            rating = row["rating"]

            q = f"有沒有 {threshold} 分以上的咖啡？"
            a = (
                f"**{name}** 品鑑評分為 {rating}/100，差 1 分未達 {threshold} 分門檻。\n\n"
                f"嚴格來說不符合「{threshold} 分以上」的條件。"
                f"但如果你願意放寬 1 分，這款咖啡的品質也相當不錯。"
            )
            pairs.append(make_conversation(q, a))

    return pairs


def generate_cot_qa(df):
    """生成推理型 Chain-of-Thought 樣本，讓模型學會逐步驗證條件"""
    pairs = []

    # CoT 類型 1：多條件篩選
    for roast_cn in ["淺焙", "中焙", "深焙"]:
        for threshold in [90, 93, 95]:
            matched = df[
                (df["rating"] >= threshold)
                & (df["roast"].map(lambda x: translate_roast(x)) == roast_cn)
            ]
            if len(matched) == 0:
                continue
            row = matched.sample(n=1, random_state=42).iloc[0]
            name = str(row.get("name", "")).strip()
            roaster = str(row.get("roaster", "")).strip()
            origin = translate_origin(row.get("origin_1", ""))
            rating = row["rating"]
            descs = []
            for col in ["desc_1", "desc_2", "desc_3"]:
                val = row.get(col, "")
                if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
                    descs.append(val.strip())
            flavors = extract_flavors(" ".join(descs))
            flavor_str = "、".join(flavors[:6]) if flavors else "均衡醇和"
            brew_tip = generate_contextual_brew_tip(roast_cn, flavors)

            q = f"推薦一款 {threshold} 分以上的{roast_cn}咖啡。"
            a = (
                f"讓我根據你的條件逐一篩選：\n\n"
                f"1. **烘焙度**：{roast_cn} → 篩選{roast_cn}咖啡\n"
                f"2. **評分門檻**：≥ {threshold} 分 → 品鑑評分需達 {threshold} 以上\n\n"
                f"符合條件的推薦：**{name}**（{roaster}）\n"
                f"- 產地：{origin}\n"
                f"- 烘焙度：{roast_cn} ✅\n"
                f"- 品鑑評分：{rating}/100 ✅（≥ {threshold}）\n"
                f"- 風味：{flavor_str}\n\n"
                f"【沖煮建議】\n{brew_tip}"
            )
            pairs.append(make_conversation(q, a))

    # CoT 類型 2：含產區的多條件
    for origin_cn in ["衣索比亞", "哥倫比亞", "肯亞", "巴西"]:
        matched = df[
            (df["rating"] >= 90)
            & (df["origin_1"].map(lambda x: translate_origin(x)) == origin_cn)
        ]
        if len(matched) == 0:
            continue
        row = matched.sample(n=1, random_state=42).iloc[0]
        name = str(row.get("name", "")).strip()
        roaster = str(row.get("roaster", "")).strip()
        roast = translate_roast(row.get("roast", ""))
        rating = row["rating"]
        descs = []
        for col in ["desc_1", "desc_2", "desc_3"]:
            val = row.get(col, "")
            if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
                descs.append(val.strip())
        flavors = extract_flavors(" ".join(descs))
        flavor_str = "、".join(flavors[:6]) if flavors else "均衡醇和"
        brew_tip = generate_contextual_brew_tip(roast, flavors)

        q = f"有沒有 90 分以上的{origin_cn}咖啡？"
        a = (
            f"讓我根據你的條件篩選：\n\n"
            f"1. **產區**：{origin_cn}\n"
            f"2. **評分門檻**：≥ 90 分\n\n"
            f"符合條件的推薦：**{name}**（{roaster}）\n"
            f"- 產地：{origin_cn} ✅\n"
            f"- 烘焙度：{roast}\n"
            f"- 品鑑評分：{rating}/100 ✅（≥ 90）\n"
            f"- 風味：{flavor_str}\n\n"
            f"【沖煮建議】\n{brew_tip}"
        )
        pairs.append(make_conversation(q, a))

    return pairs


def generate_general_knowledge_qa():
    """生成通用咖啡知識問答（不依賴單一咖啡品項）"""
    qa_list = [
        ("什麼是精品咖啡？",
         "精品咖啡（Specialty Coffee）是指經過嚴格品質把關，品鑑評分達到 80 分以上（滿分 100）的咖啡。"
         "與商業咖啡不同，精品咖啡強調：\n\n"
         "1. **產地可追溯性**：明確標示產區、莊園、處理法\n"
         "2. **品質導向**：從種植、採收、處理到烘焙，每個環節都追求卓越\n"
         "3. **風味多樣性**：展現產區獨特的地域之味（Terroir）\n"
         "4. **新鮮度**：注重烘焙日期，建議在烘焙後 7-30 天內飲用"),

        ("淺焙和深焙咖啡有什麼不同？",
         "淺焙與深焙的主要差異：\n\n"
         "| 項目 | 淺焙 | 深焙 |\n"
         "|------|------|------|\n"
         "| 顏色 | 淺棕色 | 深褐至黑色 |\n"
         "| 酸質 | 明亮、活潑 | 低、圓潤 |\n"
         "| 醇厚度 | 輕盈 | 厚實 |\n"
         "| 風味 | 花香、果酸、茶感 | 巧克力、堅果、焦糖 |\n"
         "| 咖啡因 | 稍高 | 稍低 |\n"
         "| 適合沖煮 | 手沖、愛樂壓 | 義式濃縮、法式濾壓 |\n\n"
         "淺焙保留更多產區風味，深焙則展現烘焙風味。選擇取決於個人偏好！"),

        ("手沖咖啡的基本步驟是什麼？",
         "手沖咖啡基本步驟：\n\n"
         "1. **研磨**：中細研磨（類似砂糖粗細），建議現磨\n"
         "2. **水溫**：88-93°C（淺焙偏高，深焙偏低）\n"
         "3. **粉水比**：1:15（例如 15g 咖啡粉配 225ml 水）\n"
         "4. **悶蒸**：注入約粉量 2 倍的水，等待 30 秒讓咖啡粉排氣\n"
         "5. **注水**：以同心圓方式緩慢注水，分 2-3 次注入\n"
         "6. **總萃取時間**：約 2:30-3:30 分鐘\n\n"
         "小提示：使用新鮮烘焙的咖啡豆（烘焙後 7-21 天）效果最佳！"),

        ("什麼是咖啡的「風味輪」？",
         "咖啡風味輪（Coffee Taster's Flavor Wheel）是由美國精品咖啡協會（SCA）制定的標準化風味描述工具。\n\n"
         "風味輪從中心向外展開：\n"
         "- **第一層**（內圈）：大類別 — 花香、果香、甜感、堅果/可可、香料等\n"
         "- **第二層**：細分類別 — 莓果、柑橘、熱帶水果等\n"
         "- **第三層**（外圈）：具體描述 — 藍莓、檸檬、芒果等\n\n"
         "品鑑咖啡時，可以參考風味輪來描述你所感受到的風味，"
         "幫助你更精準地了解自己的口味偏好。"),

        ("新手應該從什麼咖啡開始嘗試？",
         "給咖啡新手的建議：\n\n"
         "1. **從中焙開始**：口感均衡，不會太酸也不會太苦\n"
         "2. **推薦產區**：\n"
         "   - 🇧🇷 巴西：堅果、巧克力調，溫和好入口\n"
         "   - 🇨🇴 哥倫比亞：均衡、焦糖甜感\n"
         "   - 🇪🇹 衣索比亞：花香果酸，咖啡的發源地\n"
         "3. **沖煮方式**：先從手沖或法式濾壓壺開始，設備簡單\n"
         "4. **循序漸進**：習慣後再嘗試淺焙（酸質明亮）或深焙（濃郁厚實）\n\n"
         "最重要的是：找到自己喜歡的風味，沒有絕對的好壞之分！"),

        ("什麼是咖啡的處理法？日曬、水洗有什麼不同？",
         "咖啡處理法是指咖啡果實採收後，去除果肉取出咖啡豆的方式，對風味影響極大：\n\n"
         "**日曬處理（Natural）**\n"
         "- 整顆咖啡果實在陽光下乾燥\n"
         "- 風味：濃郁果香、酒感、甜度高、body 厚實\n"
         "- 代表產區：衣索比亞、巴西\n\n"
         "**水洗處理（Washed）**\n"
         "- 先去除果肉，再浸泡發酵去除果膠\n"
         "- 風味：乾淨、明亮酸質、風味清晰\n"
         "- 代表產區：哥倫比亞、肯亞\n\n"
         "**蜜處理（Honey）**\n"
         "- 保留部分果膠進行乾燥\n"
         "- 風味：介於日曬和水洗之間，甜感佳\n"
         "- 代表產區：哥斯大黎加"),
    ]

    def add_variants(questions, answer):
        for question in questions:
            qa_list.append((question, answer))

    # --- 產區比較與產區解釋 ---
    for origin, profile in ORIGIN_PROFILES.items():
        answer = (
            f"{origin}咖啡的常見特色可以從三個角度理解：\n\n"
            f"1. **風味主軸**：{profile['style']}。\n"
            f"2. **口感結構**：{profile['body']}。\n"
            f"3. **適合族群**：{profile['best_for']}。\n\n"
            "實際風味仍會受到品種、海拔、處理法與烘焙度影響，所以產區特色應視為判斷方向，"
            "不是絕對規則。"
        )
        add_variants(
            [
                f"{origin}咖啡通常有什麼特色？",
                f"請介紹{origin}產區咖啡的風味。",
                f"{origin}咖啡適合什麼口味的人？",
            ],
            answer,
        )

    origin_pairs = [
        ("衣索比亞", "哥倫比亞"),
        ("衣索比亞", "肯亞"),
        ("哥倫比亞", "巴西"),
        ("哥倫比亞", "瓜地馬拉"),
        ("巴西", "哥斯大黎加"),
        ("瓜地馬拉", "巴拿馬"),
        ("肯亞", "巴西"),
        ("印尼", "衣索比亞"),
        ("哥斯大黎加", "巴拿馬"),
        ("印尼", "哥倫比亞"),
    ]
    for left, right in origin_pairs:
        left_profile = ORIGIN_PROFILES[left]
        right_profile = ORIGIN_PROFILES[right]
        answer = (
            f"{left}和{right}的差異可以這樣比較：\n\n"
            f"| 面向 | {left} | {right} |\n"
            f"|------|------|------|\n"
            f"| 常見風味 | {left_profile['style']} | {right_profile['style']} |\n"
            f"| 口感結構 | {left_profile['body']} | {right_profile['body']} |\n"
            f"| 適合偏好 | {left_profile['best_for']} | {right_profile['best_for']} |\n\n"
            "如果你喜歡更鮮明的酸質和香氣，可以優先選風味較明亮的一方；"
            "如果想要穩定、圓潤、低刺激，則可以選擇口感較厚實或均衡的產區。"
        )
        add_variants(
            [
                f"{left}和{right}的咖啡有什麼不同？",
                f"請比較{left}與{right}咖啡的風味差異。",
                f"新手要怎麼在{left}和{right}咖啡之間選擇？",
            ],
            answer,
        )

    # --- 烘焙度、沖煮參數與萃取診斷 ---
    for roast, profile in ROAST_PROFILES.items():
        answer = (
            f"{roast}咖啡的重點如下：\n\n"
            f"1. **風味表現**：{profile['flavor']}。\n"
            f"2. **常見風險**：{profile['risk']}。\n"
            f"3. **沖煮建議**：{profile['brew']}\n\n"
            "調整時建議一次只改一個變因，例如水溫、研磨度或粉水比，"
            "比較容易判斷味道變化的來源。"
        )
        add_variants(
            [
                f"{roast}咖啡應該怎麼沖比較好喝？",
                f"{roast}適合什麼沖煮參數？",
                f"{roast}咖啡的風味和沖煮重點是什麼？",
                f"我買了{roast}咖啡，水溫和研磨要怎麼抓？",
            ],
            answer,
        )

    brew_issue_answers = [
        (
            ["咖啡喝起來太酸怎麼調整？", "手沖太酸是什麼原因？", "咖啡酸到不舒服要怎麼修正？"],
            "咖啡太酸通常代表萃取不足或豆子本身酸質較高。你可以依序嘗試：\n\n"
            "1. **磨細一點**：增加接觸面積，提高萃取率。\n"
            "2. **提高水溫**：例如從 88°C 調到 91°C。\n"
            "3. **延長萃取時間**：注水放慢或增加悶蒸後的總時間。\n"
            "4. **確認烘焙度**：淺焙本來酸質較明亮，若不喜歡酸可改中焙或中深焙。\n\n"
            "建議一次只調一個變因，避免不知道是哪個調整造成改善。"
        ),
        (
            ["咖啡喝起來太苦怎麼辦？", "手沖太苦要怎麼調？", "為什麼我的咖啡有焦苦味？"],
            "咖啡太苦常見原因是過度萃取、水溫太高、研磨太細或深焙豆本身苦味較重。可以嘗試：\n\n"
            "1. **磨粗一點**：降低萃取率。\n"
            "2. **降低水溫**：中深焙或深焙可落在 85-89°C。\n"
            "3. **縮短萃取時間**：避免尾段萃出過多苦澀物質。\n"
            "4. **調整粉水比**：若太濃，可從 1:14 調到 1:15 或 1:16。\n\n"
            "如果苦味伴隨乾澀，通常更接近過度萃取；如果只是厚實低酸，可能是烘焙風格。"
        ),
        (
            ["咖啡喝起來很薄怎麼改善？", "手沖咖啡水水的是什麼原因？", "咖啡沒有甜感和厚度怎麼辦？"],
            "咖啡喝起來薄，通常是萃取不足或濃度不足。可以從這幾點調整：\n\n"
            "1. **提高粉量或降低水量**：例如從 1:16 改成 1:15。\n"
            "2. **磨細一點**：提高萃取率與口感厚度。\n"
            "3. **延長萃取時間**：讓甜感和 body 更完整。\n"
            "4. **檢查豆子新鮮度**：太舊的豆子香氣和甜感會下降。\n\n"
            "若已經有苦味但仍覺得薄，可能是萃取不均，需要改善注水穩定度。"
        ),
        (
            ["咖啡有澀感是什麼原因？", "為什麼咖啡喝完口腔很乾？", "手沖出現乾澀感要怎麼修？"],
            "乾澀感通常和過度萃取、細粉過多或注水擾動太強有關。可以嘗試：\n\n"
            "1. **磨粗一點**，減少尾段苦澀物質。\n"
            "2. **降低水溫**，尤其是中深焙或深焙。\n"
            "3. **減少攪拌與大水流衝擊**，避免細粉堵塞濾紙。\n"
            "4. **縮短尾段萃取**，不要讓最後很慢的水流全部滴完。\n\n"
            "澀感和苦味不同：苦味是味覺，澀感更像口腔被收乾的觸感。"
        ),
    ]
    for questions, answer in brew_issue_answers:
        add_variants(questions, answer)

    # --- 處理法比較 ---
    for process, profile in PROCESS_PROFILES.items():
        answer = (
            f"{process}處理的重點如下：\n\n"
            f"1. **處理方式**：{profile['method']}\n"
            f"2. **常見風味**：{profile['flavor']}\n"
            f"3. **注意事項**：{profile['risk']}\n\n"
            "處理法會影響風味，但不是唯一因素；產區、品種、海拔、烘焙度也會一起決定最後杯中表現。"
        )
        add_variants(
            [
                f"什麼是{process}處理？",
                f"{process}處理的咖啡喝起來有什麼特色？",
                f"{process}處理適合什麼口味的人？",
            ],
            answer,
        )

    process_pairs = [
        ("日曬", "水洗"),
        ("日曬", "蜜處理"),
        ("水洗", "蜜處理"),
        ("厭氧發酵", "水洗"),
        ("厭氧發酵", "日曬"),
    ]
    for left, right in process_pairs:
        left_profile = PROCESS_PROFILES[left]
        right_profile = PROCESS_PROFILES[right]
        answer = (
            f"{left}和{right}的差異可以這樣看：\n\n"
            f"| 面向 | {left} | {right} |\n"
            f"|------|------|------|\n"
            f"| 處理方式 | {left_profile['method']} | {right_profile['method']} |\n"
            f"| 常見風味 | {left_profile['flavor']} | {right_profile['flavor']} |\n"
            f"| 風險 | {left_profile['risk']} | {right_profile['risk']} |\n\n"
            "如果你喜歡乾淨、清楚、酸質明亮，通常可先選水洗；"
            "如果你喜歡濃郁果香或特殊發酵感，可以嘗試日曬、蜜處理或厭氧發酵。"
        )
        add_variants(
            [
                f"{left}和{right}處理法有什麼不同？",
                f"請比較{left}與{right}咖啡的風味差異。",
                f"新手要怎麼在{left}和{right}之間選擇？",
            ],
            answer,
        )

    # --- 評分標準與高分咖啡 ---
    score_answer = (
        "咖啡評分通常不是只看單一風味，而是綜合香氣、酸質、甜感、body、平衡感、乾淨度、尾韻與整體印象。\n\n"
        "- **80 分以上**：通常可被視為精品咖啡門檻。\n"
        "- **90 分以上**：風味清楚、缺陷少，整體品質很高。\n"
        "- **95 分以上**：通常需要非常突出的香氣、層次、甜感、乾淨度與稀有性。\n\n"
        "要注意的是，高分不等於每個人都喜歡。喜歡低酸厚實的人，不一定會偏好高分淺焙花果香咖啡。"
    )
    add_variants(
        [
            "什麼樣的咖啡能拿到高分？",
            "咖啡評分 90 分和 95 分差在哪？",
            "95 分以上的咖啡通常有什麼特徵？",
            "咖啡評分是怎麼判斷的？",
            "高分咖啡一定比較好喝嗎？",
        ],
        score_answer,
    )

    tasting_answer = (
        "品鑑咖啡時，可以用「香氣、酸質、甜感、body、尾韻、平衡感」六個面向來描述：\n\n"
        "1. **香氣**：乾香與濕香是否有花香、果香、堅果或巧克力調。\n"
        "2. **酸質**：是明亮、柔和、尖銳，還是幾乎沒有酸感。\n"
        "3. **甜感**：像蜂蜜、焦糖、黑糖或水果甜。\n"
        "4. **body**：口感是輕盈、滑順、奶油感，還是厚實。\n"
        "5. **尾韻**：吞下後香氣停留多久，是否乾淨。\n"
        "6. **平衡感**：酸、甜、苦、香氣是否協調。\n\n"
        "初學者不用急著說得很專業，先分辨「喜歡/不喜歡」和原因，就是很好的開始。"
    )
    add_variants(
        [
            "我要怎麼描述一杯咖啡的味道？",
            "咖啡品鑑可以從哪些面向開始？",
            "新手怎麼練習咖啡風味描述？",
            "什麼是 body、酸質和尾韻？",
        ],
        tasting_answer,
    )

    selection_answer = (
        "選咖啡可以先用三個問題縮小範圍：\n\n"
        "1. **你能接受酸嗎？** 喜歡明亮酸質可選淺焙、衣索比亞、肯亞；不喜歡酸可選中焙以上、巴西或印尼。\n"
        "2. **你喜歡什麼風味？** 花果香可選淺焙水洗或日曬；巧克力堅果可選中焙或中深焙。\n"
        "3. **你用什麼器材？** 手沖適合展現層次；義式適合厚實甜感；法壓適合 body。\n\n"
        "如果不確定，建議從中焙、哥倫比亞或巴西開始，因為它們通常比較均衡、容錯率高。"
    )
    add_variants(
        [
            "新手買咖啡豆要怎麼選？",
            "我不知道自己喜歡什麼咖啡，該怎麼開始？",
            "如何根據口味挑咖啡豆？",
            "不喜歡太酸的人適合什麼咖啡？",
            "喜歡花果香的人適合什麼咖啡？",
        ],
        selection_answer,
    )

    return [make_conversation(q, a) for q, a in qa_list]


def main():
    random.seed(42)
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 嘗試讀取 CSV
    csv_candidates = [
        raw_dir / "simplified_coffee.csv",
        raw_dir / "coffee_analysis.csv",
        raw_dir / "top-rated-coffee-clean.csv",
    ]
    # 也搜尋 raw 目錄下（含子目錄）任何 CSV
    if raw_dir.exists():
        csv_candidates += list(raw_dir.glob("**/*.csv"))

    df = None
    for csv_path in csv_candidates:
        if csv_path.exists():
            print(f"[INFO] 讀取: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"[INFO] 欄位: {list(df.columns)}")
            print(f"[INFO] 筆數: {len(df)}")
            break

    if df is None:
        print("=" * 60)
        print("[ERROR] 找不到咖啡資料集 CSV 檔案！")
        print()
        print("請從 Kaggle 下載資料集：")
        print("https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset")
        print()
        print(f"下載後將 CSV 檔案放到: {raw_dir}")
        print("=" * 60)
        return

    # 確認必要欄位
    required = ["name", "roast", "rating"]
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] CSV 缺少必要欄位: {col}")
            return

    # 清洗資料
    df = df.dropna(subset=["name", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    print(f"[INFO] 清洗後筆數: {len(df)}")

    # ========== 按咖啡品項做 Train/Val/Test 切分 ==========
    # 同一款咖啡的所有 QA 對必須在同一個集合中，避免資料洩漏
    unique_names = list(df["name"].unique())
    random.shuffle(unique_names)

    n = len(unique_names)
    train_names = set(unique_names[:int(n * 0.8)])
    val_names = set(unique_names[int(n * 0.8):int(n * 0.9)])
    test_names = set(unique_names[int(n * 0.9):])

    df_train = df[df["name"].isin(train_names)]
    df_val = df[df["name"].isin(val_names)]
    df_test = df[df["name"].isin(test_names)]

    print(f"[INFO] 切分結果：Train {len(df_train)} 筆 / Val {len(df_val)} 筆 / Test {len(df_test)} 筆")
    print(f"[INFO] 咖啡品項：Train {len(train_names)} 款 / Val {len(val_names)} 款 / Test {len(test_names)} 款")

    # ========== 生成各集合的 QA 對 ==========
    def build_pairs(subset_df, include_extras=False):
        """為指定的 DataFrame 子集生成 QA 對"""
        pairs = []
        for _, row in subset_df.iterrows():
            pairs.extend(generate_qa_pairs(row))
        if include_extras:
            # 通用知識、數值邊界、CoT 只放入 train set
            pairs.extend(generate_general_knowledge_qa())
            pairs.extend(generate_score_threshold_qa(subset_df))
            pairs.extend(generate_cot_qa(subset_df))
        random.shuffle(pairs)
        return pairs

    train_pairs = build_pairs(df_train, include_extras=True)
    val_pairs = build_pairs(df_val, include_extras=False)
    test_pairs = build_pairs(df_test, include_extras=False)

    # ========== 儲存 ==========
    splits = {
        "coffee_train.json": train_pairs,
        "coffee_val.json": val_pairs,
        "coffee_test.json": test_pairs,
    }
    for filename, pairs in splits.items():
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"[儲存] {filename}: {len(pairs)} 筆")

    print(f"\n{'=' * 60}")
    print(f"[完成] 共生成 {sum(len(p) for p in splits.values())} 筆資料")
    print(f"  Train: {len(train_pairs)} 筆")
    print(f"  Val:   {len(val_pairs)} 筆")
    print(f"  Test:  {len(test_pairs)} 筆")
    print(f"[儲存] {out_dir}")
    print(f"{'=' * 60}")

    # 顯示範例
    print("\n--- 範例資料（Train）---")
    sample = random.choice(train_pairs)
    for msg in sample["conversations"]:
        role = msg["role"].upper()
        print(f"[{role}] {msg['content'][:200]}...")


if __name__ == "__main__":
    main()

