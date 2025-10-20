from collections import OrderedDict

# 知识库定义
KNOWLEDGE_BASE = [
    {"if": {"毛发": True}, "then": "哺乳动物"},
    {"if": {"奶": True}, "then": "哺乳动物"},
    {"if": {"羽毛": True}, "then": "鸟"},
    {"if": {"飞": True, "下蛋": True}, "then": "鸟"},
    {"if": {"吃肉": True}, "then": "食肉动物"},
    {"if": {"锋利牙齿": True, "锋利爪子": True, "眼盯前方": True}, "then": "食肉动物"},
    {"if": {"哺乳动物": True, "有蹄": True}, "then": "有蹄类动物"},
    {"if": {"哺乳动物": True, "反刍动物": True}, "then": "有蹄类动物"},
    {"if": {"哺乳动物": True, "食肉动物": True, "黄褐色": True, "暗斑点":True}, "then": "金钱豹"},
    {"if": {"哺乳动物": True, "食肉动物": True, "黄褐色": True, "黑色条纹": True}, "then": "虎"},
    {"if": {"有蹄类动物": True, "长脖子": True, "长腿": True, "暗斑点": True}, "then": "长颈鹿"},
    {"if": {"有蹄类动物": True, "黑色条纹": True}, "then": "斑马"},
    {"if": {"鸟": True, "长脖子": True, "长腿": True, "不会飞": True, "黑白二色": True}, "then": "鸵鸟"},
    {"if": {"鸟": True, "游泳": True, "长腿": True, "不会飞": True, "黑白二色": True}, "then": "企鹅"},
    {"if": {"鸟": True, "善飞": True}, "then": "信天翁"},
]

ALL_FEATURES = []
active_features = {}


def build_feature_list():
    feature_set = set()
    for rule in KNOWLEDGE_BASE:
        feature_set.update(rule["if"].keys())
    global ALL_FEATURES
    ALL_FEATURES = list(OrderedDict.fromkeys(feature_set))


def show_prompt():
    global ALL_FEATURES
    build_feature_list()
    print("在以下特征中，选取动物特征（输入特征前面的序号，每行输入一个特征，空行表示输入结束）：")
    for i, name in enumerate(ALL_FEATURES, 1):
        print(f"{i}: {name}", end="  ")
        if i % 8 == 0:
            print()
    print()


def gather_features():
    global ALL_FEATURES, active_features
    active_features = {f: False for f in ALL_FEATURES}
    print("请输入特征序号：")
    line = input().strip()
    while line:
        idx = int(line) - 1
        if 0 <= idx < len(ALL_FEATURES):
            active_features[ALL_FEATURES[idx]] = True
        line = input().strip()


def run_inference():
    for rule in KNOWLEDGE_BASE:
        conditions_met = True
        for condition_key, condition_value in rule["if"].items():
            if active_features.get(condition_key) != condition_value:
                conditions_met = False
                break
        if conditions_met:
            return rule["then"]
    return "未知动物"


def main():
    show_prompt()
    gather_features()
    print("\n正在识别...\n")
    result = run_inference()
    print(f"识别出的动物是：{result}")


if __name__ == "__main__":
    main()
