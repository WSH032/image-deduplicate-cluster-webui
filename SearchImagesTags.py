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
        """ 搜索目录下已注册的扩展名的所有图片，返回一个带扩展，不带路径的名字列表 """
        
        
        # 获取该目录下所有的图片文件名
        image_files_names_list = [f for f in os.listdir(self.search_dir)
                                  if os.path.isfile( os.path.join(self.search_dir, f) ) and f.lower().endswith(self._IMAGE_EXTENSION)
        ]
        return image_files_names_list
    
    
    def tag_files(self) -> list[str]:
        """ 搜索目录下对应图片名字的txt文件，返回一个仅仅改变扩展名为txt的列表 """
        
        # 读取图片列表
        images_names_list = self.image_files()
        
        # 将扩展名改为tags_file_ext
        tag_files_names_list = [ self.change_ext_with_old_name(f, self.tag_files_ext  ) for f in images_names_list ]
        return tag_files_names_list
    
    
    def tag_content(self, error_then_tag_is: str="") -> list[str]:
        """
        读取目录下对应名字的txt文件内容，返回内容str为元素的列表 
        
        error_then_tag_is为无法读取内容时候替代的内容
        """
        
        # 读取tas文件列表
        tag_files_names_list = self.tag_files()
        
        # 读取tag文件内容
        tag_content_list = []
        for f in tag_files_names_list:
            try:
                with open( os.path.join(self.search_dir, f) ) as content:
                    tag_content_list.append( content.read() )
            except Exception as e:
                logging.error(f"读取 {f} 发生错误, error: {e}")
                tag_content_list.append( error_then_tag_is )
        return tag_content_list
