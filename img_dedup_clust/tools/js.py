"""为整个项目提供涉及js的工具"""

import textwrap
from typing import Optional, Any, Literal


#################### 常量 ####################

DomAlias = Literal["document", "gradioApp()"]

DEFALUT_DOM: DomAlias = "document"
GRADIO_DOM: DomAlias = "gradioApp()"

ELEM_ID_PREFIX = "image_deduplicate_cluster_webui_"  # 用于统一本项目中gradio组件的elem_id前缀


#################### 工具类与函数 ####################

def strict_setattr(obj: Any, name: str, value: Any) -> None:
    """严格的setattr函数，用于检查是否有该属性，然后再赋值"""
    if hasattr(obj, name):
        setattr(obj, name, value)
    else:
        raise AttributeError(f"{obj} has not {name} attribute")


class BaseJS:
    """存储 _dom 和 _indent_str 两个类属性，用于生成js代码

    _dom (DomAlias): 类属性，用于指定js代码中搜索父对象为document或者gradioApp()
    _indent_str (str): 类属性，用于指定js代码中的缩进空格字符串
    """

    _dom: DomAlias  = DEFALUT_DOM
    _indent_str = " " * 4

    def __init__(self, is_a1111_webui: Optional[bool] = None, indent: Optional[int] = None):
        """初始化
        如果保持相应的Optional参数为None，则可以通过调用父类的set_cls_attr()方法，起到子类递归修改的效果
        如果不想受子类递归影响，请显式指定参数，将会覆盖 _dom 和 _indent_str 为实例属性

        Args:
            is_a1111_webui (Optional[bool], optional): 实例属性，当为真时，将会使用gradioApp()，否则使用document. Defaults to None.
            indent (Optional[int], optional): 实例属性，代码缩进空格数，默认为4. Defaults to None.
        """
        self.is_a1111_webui = is_a1111_webui
        self.indent = indent
        self.set_self_attr(is_a1111_webui, indent)

    @classmethod
    def set_cls_attr(cls, is_a1111_webui: Optional[bool] = None, indent: Optional[int] = None):
        """修改类属性
        在保证子类未重写相应属性的情况下（及所有Optional参数都为None），直接修改父类，可做到全局效果

        Args:
            is_a1111_webui (Optional[bool], optional): 当为真时，将会使用gradioApp()，否则使用document. Defaults to None.
            indent (Optional[int], optional): 代码缩进空格数，默认为4. Defaults to None.
        """
        if is_a1111_webui is not None:
            cls._dom = GRADIO_DOM if is_a1111_webui else DEFALUT_DOM
        if indent is not None:
            cls._indent_str = " " * indent

    def set_self_attr(self, is_a1111_webui: Optional[bool] = None, indent: Optional[int] = None):
        """修改实例属性

        Args:
            is_a1111_webui (Optional[bool], optional): 当为真时，将会使用gradioApp()，否则使用document. Defaults to None.
            indent (Optional[int], optional): 代码缩进空格数，默认为4. Defaults to None.
        """
        if is_a1111_webui is not None:
            self._dom = GRADIO_DOM if is_a1111_webui else DEFALUT_DOM
        if indent is not None:
            self._indent_str = " " * indent
    
    def del_self_attr(self):
        """删除实例属性 _dom 和 _indent_str，使相应属性恢复为类属性，以便于子类递归修改"""
        # 不管它是实例属性还是类属性，先访问一下，保证这个属性确实存在
        # 避免 except AttributeError: pass 隐式忽略了某些错误
        (self._dom, self._indent_str)  # type: ignore
        # 可能不存在实例属性，只存在类属性，所以用try
        try:
            del self._dom
        except AttributeError:
            pass

        try:
            del self._indent_str
        except AttributeError:
            pass


class WrapJsForReadyState(BaseJS):
    def make_js(self, js_func_str: str, js_func_name: str) -> str:
        """传入一个字符串形式的js函数，会在网页加载完成后，执行该函数

        Args:
            js_func_str (str): 字符串形式的js函数代码
            js_func_name (str): js函数名

        Returns:
            str: 字符串形式的js代码
                # 前三行为js_func_str
                function js_func_name() {
                    // 函数体
                }
                if (document.readyState === "loading") {
                    document.addEventListener("DOMContentLoaded", js_func_name);
                }
                else {
                    js_func_name();
                }
        """
        indent_str = self._indent_str
        DOM = self._dom

        readyState_js_str = textwrap.dedent(f"""\
            // 如果网页还在加载，则等待网页加载完成后执行
            if ({DOM}.readyState === "loading") {{
            {indent_str}{DOM}.addEventListener("DOMContentLoaded", {js_func_name});
            }}
            // 如果已经加载完成，则直接执行
            else {{
            {indent_str}{js_func_name}();
            }}
        """)

        js_str = js_func_str + "\n" + readyState_js_str

        return js_str 


# TODO: 把它修改为类，继承BaseJS
def wrap_js_to_function(js_str: str, func_name: str="", indent: int=4) -> str:
    """将js代码包装成一个函数

    Args:
        js_str (str): js代码
        func_name (str, optional): 函数名，留空则为带括号的匿名函数表达式. Defaults to "".
        indent (int, optional): 缩进. Defaults to 4.

    Returns:
        str: 包装后的js代码

            func_name不留空:
                function func_name() {
                    // 函数体
                }

            func_name留空:
                (function() {
                    // 函数体
                })
    """

    indent_str = " " * indent
    # 先将js_str除了第一行外，每一行进行缩进{indent}个空格
    js_str = textwrap.indent(js_str, indent_str)
    
    # 注意{js_str}的位置不要再缩进了，同时不要用dedent，因为{js_str}内的缩进与外面的缩进不一样，dedent不会起作用

    # function func_name() {
    #     // 函数体
    # }
    if func_name:
        js_function_str = f"""\
function {func_name}() {{
{js_str}
}}
"""
    
    # (function() {
    #     // 函数体
    # })
    else:
        js_function_str = f"""\
(function() {{
{js_str}
}})
"""

    return js_function_str
