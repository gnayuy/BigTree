# BigTree

BigTree, Big ImaGe hierarchical TREE construction software, is developed to efficient reformat large-scale images data sets into a hierarchical tree data structure that can be visualize and annotate with Vaa3D-TeraFly [1][].

More details see our paper "BigTree: high-performance hierarchical tree construction for large image data sets".

## compile with cmake

Install [libtiff][] by following the instructions on their website. This is required for read and write images.

We use [cxxopts][] to parse command lines, which requires gcc > 4.9.

In terminal, type commands:

    % mkdir build
    % cd build
    % ccmake ..
    % make

## use BigTree

Assuming your input images are 2D image slices stored in one folder in order. For example, converting to a 5-scale tree data structure, we simply type command:

    % BigTree -i <path_dir_input_images> -o <path_dir_output> -n 5
    
This is equavelent to using [TeraConverter][] as:

    % teraconverter -s="<path_dir_input_images>" -d="<path_dir_output>" --sfmt="TIFF (series, 2D)" --dfmt="TIFF (tiled, 3D)" --rescale=4 --resolutions=01234 --width=256 --height=256 --depth=256 --halve=max --libtiff_rowsperstrip=-1
    
## image file formats

[TIFF][]: tiff file format

## references

[1]. Bria, A., Iannello, G., Onofri, L., and Peng, H. (2016). TeraFly: real-time threedimensional visualization and annotation of terabytes of multidimensional volumetric images. Nat. Methods 13, 192–194. doi: 10.1038/nmeth.3767

[libtiff]:http://www.libtiff.org
[TIFF]:http://www.libtiff.org/support.html
[cxxopts]:https://github.com/jarro2783/cxxopts
[TeraConverter]:https://github.com/Vaa3D/Vaa3D_Wiki/wiki/TeraConverter
