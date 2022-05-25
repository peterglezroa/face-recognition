"""
visualization.py
----------------
Visualization methods to view images and clusters
"""

def view_cluster_as_image(cluster:list, max_imgs:int=8) -> None:
    """
    def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

get_concat_h(im1, im1).save('data/dst/pillow_concat_h.jpg')
get_concat_v(im1, im1).save('data/dst/pillow_concat_v.jpg')
"""
