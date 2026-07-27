
"""对要删除的商品名字段做转义处理，防止delete字符串表达式转化失败"""
def escape_milvus_string(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    s = (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\r", " ")
         .replace("\n", " ")
         )
    return s