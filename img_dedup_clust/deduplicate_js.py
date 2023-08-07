import textwrap

from img_dedup_clust.tools.js import (
    BaseJS,
    WrapJsForReadyState,
)


class Keydown2Click(BaseJS):
    def make_js(
        self,
        obj_id: str,
        button_id: str,
        keyboard_key: str,
    ) -> str:
        """生成字符串形式的js代码，用于在网页加载完成后，为指定的js对象添加键盘事件，按下指定的键盘按键时，触发指定按钮的点击事件

        Args:
            obj_id (str): 指定的js对象的elem_id
            button_id (str): 所要触发点击事件的按钮的elem_id
            keyboard_key (str): 键盘按键，如"Enter", "Escape"

        Returns:
            str: 字符串形式的js代码
        """    
        DOM = self._dom
        js_str = textwrap.dedent(f"""\
            function keydown2click(event){{
                // 当按下{keyboard_key}键，则触发{button_id}按钮的点击事件
                const btn = {DOM}.getElementById("{button_id}");
                if (!btn) return;
                if (event.key === "{keyboard_key}") btn.click();
            }}

            function addEventListener(){{
                // 为{obj_id}添加键盘事件
                {DOM}.getElementById("{obj_id}").addEventListener('keydown', keydown2click, false);
            }}
        """)
        wrap_js_for_readyState = WrapJsForReadyState(self.is_a1111_webui, self.indent)
        js_str = wrap_js_for_readyState.make_js(js_str, "addEventListener")

        return js_str


class Click2Hide(BaseJS):
    def make_js(
        self,
        obj_id: str,
    ) -> str:
        """生成字符串形式的js代码，用于在网页加载完成后，为指定的js对象添加点击事件，点击该对象时，将其隐藏

        Args:
            obj_id (str): 指定的js对象的elem_id

        Returns:
            str: 字符串形式的js代码
        """    
        DOM = self._dom
        js_str = textwrap.dedent(f"""\
            function addEventListener(){{
                const elem = {DOM}.getElementById('{obj_id}');
                // 为{obj_id}添加点击事件，点击时隐藏其自身
                elem.addEventListener('click', function() {{
                    elem.style.display = 'none';
                }});
            }}
        """)
        wrap_js_for_readyState = WrapJsForReadyState(self.is_a1111_webui, self.indent)
        js_str = wrap_js_for_readyState.make_js(js_str, "addEventListener")

        return js_str
