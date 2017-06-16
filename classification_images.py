from toydata_base import *

class image_gen_counter:
    _counter_ = 0
def make_image_library(num_images=10,debug=0,bad_label=False):
    """
    This function makes a set of classification images, labeled with shape type  i\
n an array of length 4. The bad_label functionality randomizes the labels assigned\
 to each image, to test training. """

    locations = []
    bad_locations = []
    images = []

    for i in range(num_images):

        if debug:
            print 'Generating image',i

        mat = np.zeros([28,28]).astype(np.float32)
        add_shapes_to(mat, locations)

        if debug>1:
            image(mat)
            plt.savefig('image_%04d.png' % image_gen_counter._counter_)

        mat = np.reshape(mat, (784))
        images.append(mat)

        image_gen_counter._counter_ +=1

    if bad_label:
        for loc in locations:
            bad_locations.append(randomize_labels())

    if bad_label:
        return images, bad_locations
    return images, locations
