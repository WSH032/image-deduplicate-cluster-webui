# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:48:40 2023

@author: WSH
"""

# 导入所需的库
import os
import logging


IMAGE_EXTENSION = ('.apng',
                    '.blp',
                    '.bmp',
                    '.bufr',
                    '.bw',
                    '.cur',
                    '.dcx',
                    '.dds',
                    '.dib',
                    '.emf',
                    '.eps',
                    '.fit',
                    '.fits',
                    '.flc',
                    '.fli',
                    '.fpx',
                    '.ftc',
                    '.ftu',
                    '.gbr',
                    '.gif',
                    '.grib',
                    '.h5',
                    '.hdf',
                    '.icb',
                    '.icns',
                    '.ico',
                    '.iim',
                    '.im',
                    '.j2c',
                    '.j2k',
                    '.jfif',
                    '.jp2',
                    '.jpc',
                    '.jpe',
                    '.jpeg',
                    '.jpf',
                    '.jpg',
                    '.jpx',
                    '.mic',
                    '.mpeg',
                    '.mpg',
                    '.mpo',
                    '.msp',
                    '.palm',
                    '.pbm',
                    '.pcd',
                    '.pcx',
                    '.pdf',
                    '.pgm',
                    '.png',
                    '.pnm',
                    '.ppm',
                    '.ps',
                    '.psd',
                    '.pxr',
                    '.ras',
                    '.rgb',
                    '.rgba',
                    '.sgi',
                    '.tga',
                    '.tif',
                    '.tiff',
                    '.vda',
                    '.vst',
                    '.webp',
                    '.wmf',
                    '.xbm',
                    '.xpm'
)



class SearchImagesTags(object):
    
    def __init__(self, search_dir: str, tag_file_ext: str=".txt"):
        self.search_dir = search_dir  # 要搜索的路径
        self.tag_files_ext = tag_file_ext  # 对应的tags文件的扩展名
        self._IMAGE_EXTENSION = IMAGE_EXTENSION  # 接受的图片扩展名
        self.image_files_list = None  # 图片文件名列表，调用image_files()方法时生成或刷新
        self.tag_files_list = None  # tags文件名列表，调用tag_files()方法时生成或刷新
        self.tag_content_list = None  # tags文件内容列表，调用tag_content()方法时生成或刷新
    
    
    @staticmethod
    def change_ext_with_old_name(path: str, new_ext: str) -> str:
        """ 将path的扩展名改成new_ext """
        path_without_ext, ext = os.path.splitext(path)
        new_path = path_without_ext + new_ext
        return new_path
    
    @staticmethod
    def change_name_with_old_ext(path: str, new_name: str) -> str:
        """ 保留path扩展名，更改其文件名为new_name """
        path_without_ext, ext = os.path.splitext(path)
        # 取出除最后一个部分的路径，将其与新名字join一起
        new_path = os.path.join( os.path.dirname(path_without_ext), new_name + ext )
        return new_path
    
    
    def image_files(self) -> list[str]:
        """ 搜索目录下已注册的扩展名的所有图片，返回不带路径的图片名字列表 """
        
        # 获取该目录下所有的图片文件名
        image_files_names_list = [f for f in os.listdir(self.search_dir)
                                  if os.path.isfile( os.path.join(self.search_dir, f) ) and f.lower().endswith(self._IMAGE_EXTENSION)
        ]
        self.image_files_list = image_files_names_list
        return self.image_files_list
    
    
    def tag_files(self) -> list[str]:
        """ 返回名字为self.image_files_list中所有图片名字，扩展名被改为 tags_file_ext 的字符串列表 """
        
        # 先判断一次，因为用户可能自行修改了self.image_files_list，不要覆盖了
        if self.image_files_list is None:
            self.image_files()
        assert self.image_files_list is not None, "在tag_files()中调用image_files()后self.image_files_list不应该为None"

        image_files_list = self.image_files_list

        # 将扩展名改为tags_file_ext
        tag_files_list = [ self.change_ext_with_old_name(f, self.tag_files_ext  ) for f in image_files_list ]
        self.tag_files_list = tag_files_list
        return self.tag_files_list
    
    
    def tag_content(self, error_then_tag_is: str="") -> list[str]:
        """
        读取由 tag_files() 返回值所指定文件的内容
        
        error_then_tag_is为无法读取内容时候替代的内容

        返回列表中元素为每个文件的内容
        """
        
        # 这里不要在判断None来决定是否调用了，强制调用一次以更新self.tag_files_list，保证对应最新的self.image_files_list
        self.tag_files()
        assert self.tag_files_list is not None, "在tag_content()中调用tag_files_list()后self.tag_files_list不应该为None"

        # 读取tas文件列表
        tag_files_list = self.tag_files_list
        
        # 读取tag文件内容
        tag_content_list = []
        for f in tag_files_list:
            try:
                with open( os.path.join(self.search_dir, f) ) as content:
                    tag_content_list.append( content.read() )
            except Exception as e:
                logging.error(f"读取 {f} 发生错误, error: {e}")
                tag_content_list.append( error_then_tag_is )
        
        self.tag_content_list = tag_content_list
        return self.tag_content_list
