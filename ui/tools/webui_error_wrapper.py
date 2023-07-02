import logging
from typing import Callable

""" (inspect.signature(func).parameters) 也可以获取输入数量 """

def webui_error_default_wrapper( args_tuple: tuple=() ) -> Callable:
    """
    用于处理WebUI中的异常,使其在发生异常时仍能返回值

    用法：
    1、
    当需要指定异常发生时的自定义的返回值时，请以tuple的形式传入，如：
    @webui_error_default_wrapper((0, 0, 0))

    2、 （这个方法需要原函数的支持）
    或者，如果在原函数的输入中，已经以*args的形式传入了上一次输出组件的值，
    则可以不用传入参数，异常时候将会把上一次输出的组件作为返回值，例子：
    output_list = [ gr.Textbox() ]
    gr.Button().click(fn=func,
                    inputs=[ gr.Textbox() ] + output_list,
                    outputs=output_list,
                )
    @webui_error_default_wrapper()
    def func(a, *args):
        # 注意，要求原函数中最后的输入参数为*args
        # 其中，输入原函数的*args的参数为需要返回值的组件先前的值
        ...
    
    特别注意，有些gradio组件是不能做为输入的； 有可能出现原先组件的值不能做为组件的返回值；比如gradio.Gallery组件
    """
    
    def decorator(func: Callable) -> Callable:
        if args_tuple:
            logging.debug(f"{func.__name__}函数的异常时返回值被指定为{args_tuple}")
        def wrapper(*wrapper_args):
            try:
                return func(*wrapper_args)
            except Exception as e:
                logging.exception(f"{func.__name__}函数出现了异常: {e}")
                # 如果最外层有输入参数，就返回最外层的输入参数
                if args_tuple:
                    return args_tuple
                # 如果没有就只能从wrapper_args中获取了
                else:
                    # func.__code__.co_argcount为func函数固定参数的个数
                    return wrapper_args[func.__code__.co_argcount:]  # 后面、包括自身，即为先前输入组件的值

        return wrapper
    
    return decorator
