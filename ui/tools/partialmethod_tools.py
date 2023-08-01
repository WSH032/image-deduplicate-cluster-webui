from typing import List, Dict, Any
import functools


def make_cls_partialmethod(
    parent_cls: type,
    cls_methdod_str_list: List[str],
    cls_method_kwargs_list: List[ Dict[str, Any] ],
) -> type:
    """继承一个父类，并把子类中指定的方法通过functools.partialmethod修改为预设参数的偏方法

    Args:
        parent_cls (type): 所需要继承的父类
        cls_methdod_str_list (List[str]): 所需要修改的方法的名字字符串列表
        cls_method_kwargs_list (List[ Dict[str, Any] ]): 对于各方法的预设参数字典

    Returns:
        type: 修改成偏方法后继承的子类

    Examples:
        class Parent:
            def my_print(self, a, b):
                print(a, b)

        Parent = make_cls_partialmethod(Parent, ["my_print"], [dict(b=1)])
        Parent().my_print(2)
    """

    cls_partialmethod_list = []  # 存放修改好的偏方法
    for cls_methdod_str, cls_method_kwargs in zip(cls_methdod_str_list, cls_method_kwargs_list):
        # 获取原始父类方法
        original_parent_cls_method = getattr(parent_cls, cls_methdod_str)
        # 修改原始父类方法为新的偏方法
        new_partialmethod = functools.partialmethod(original_parent_cls_method, **cls_method_kwargs)
        # 存放
        cls_partialmethod_list.append( (cls_methdod_str, new_partialmethod) )

    # 继承父类
    class PartialmethodCls(parent_cls):
        def __str__(self):
            return f"PartialmethodCls: {parent_cls.__name__}"
        
        def __repr__(self):
            return self.__str__()

    # 修改继承的子类中方法成相应的偏方法
    for cls_methdod_str, new_partialmethod in cls_partialmethod_list:
        setattr( PartialmethodCls, cls_methdod_str, new_partialmethod )

    # 返回修改好的子类
    return PartialmethodCls
