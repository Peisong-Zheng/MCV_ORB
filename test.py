import argparse

# 1. 创建解析器
parser = argparse.ArgumentParser(description="分析气候数据")

# 2. 定义接受哪些参数
parser.add_argument("input_file", help="输入 CSV 文件路径")

parser.add_argument(
    "--window",
    type=int,
    default=500,
    help="窗口长度，默认 500",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="开启详细输出",
)

# 3. 解析命令行参数
args = parser.parse_args()

# 4. 使用解析结果
print("输入文件：", args.input_file)
print("窗口长度：", args.window)

if args.verbose:
    print("详细输出已开启")