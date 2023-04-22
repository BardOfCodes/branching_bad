from .parser import CSG2DParser
from .compiler import CSG2DCompiler


class CSG2DExecutor:

    def __init__(self, config, device):

        self.resolution = config.RESOLUTION
        self.parser = CSG2DParser(device)
        self.compiler = CSG2DCompiler(self.resolution, device)

    def compile(self, expression):
        parsed_graphs, draw_count = self.parser.parse(expression)
        draw_transforms, inversion_array, intersection_matrix = self.compiler.fast_sub_compile(
            parsed_graphs, draw_count)
        return draw_transforms, inversion_array, intersection_matrix

    def execute(self, draw_transforms, inversion_array, intersection_matrix):

        canvas = self.compiler.evaluate(
            draw_transforms, inversion_array, intersection_matrix)
        return canvas

    def set_device(self, device):
        self.parser.set_device(device)
        self.compiler.set_device(device)