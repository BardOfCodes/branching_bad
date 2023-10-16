import networkx as nx
import random
import time
import sympy as sp
import pyeda
import cProfile
import functools
import torch as th
import numpy as np
from collections import defaultdict
from .transforms import Transforms2D
from .utils import get_points


CMD_STR_MAP = dict()
draw_seq = ["cuboid", "sphere", "cylinder"]
for item in draw_seq:
    cur_dict = {item + "_" + str(x): (item, x) for x in range(20)}
    CMD_STR_MAP.update(cur_dict)


INVERTED_MAP = {
    "union": "intersection",
    "intersection": "union",
    "difference": "union",
}
NORMAL_MAP = {
    "union": "union",
    "intersection": "intersection",
    "difference": "intersection",
}


ONLY_SIMPLIFY_RULES = set(["ii", "uu"])
ALL_RULES = set(["ii", "uu", "iu"])


class CSG2DCompiler:
    def __init__(self, resolution, device):

        self.transforms = Transforms2D(device)
        self.resolution = resolution
        self.device = th.device("cuda")
        self.draw_seq = ["cuboid", "sphere"]
        self.neg_inf = th.FloatTensor([-th.inf]).to(self.device)

        self.transform_to_execute = {
            'translate': self.transforms.get_affine_translate,
            'rotate': self.transforms.get_affine_rotate,
            'scale': self.transforms.get_affine_scale,
        }
        self.draw_to_execute = {
            'sphere': self.sphere,
            'cuboid': self.cuboid,
        }

    def set_device(self, device):
        self.device = device
        self.neg_inf = self.neg_inf.to(self.device)
        self.transforms.set_device(device)

    def sphere(self, points):
        """Return SDF at points wwith a sphere centered at origin.

        Args:
            points (torch.Tensor): B, N, 3
        """
        base_sdf = points.norm(dim=-1)
        base_sdf = base_sdf - 0.5
        return base_sdf

    def cuboid(self, points):
        """Return SDF at points with a unit cuboid centered at origin.

        Args:
            points (torch.Tensor): B, N, 3
        """
        points = th.abs(points)
        points[..., 0] -= 0.5
        points[..., 1] -= 0.5
        base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
            th.clip(th.amax(points, -1), max=0)
        return base_sdf

    def get_rule_match(self, command_graph, only_simplify=False):
        if only_simplify:
            rule_strings = ONLY_SIMPLIFY_RULES  # ["i->i", "u->u"]
        else:
            rule_strings = ALL_RULES  # ["i->i", "u->u", "i->u"]
        node_list = [command_graph.nodes[1]]
        rule_match = None
        while node_list:
            cur_node = node_list.pop()
            node_id = cur_node['id']
            command = cur_node['command']
            c_type = command['type']
            children = [command_graph.nodes[child_id]
                        for child_id in command_graph.successors(node_id)]
            node_list.extend(children[::-1])
            # op:
            if c_type == "B":
                for child in children:
                    cur_string = cur_node['command']['symbol'][0] + \
                        child['command']['symbol'][0]
                    if cur_string in rule_strings:
                        rule_match = (cur_node['id'], child['id'], cur_string)
            if rule_match:
                break

        return rule_match

    def resolve_rule(self, command_graph, resolve_rule):
        node_a = command_graph.nodes[resolve_rule[0]]
        node_b = command_graph.nodes[resolve_rule[1]]
        match_type = resolve_rule[2]
        if match_type in ONLY_SIMPLIFY_RULES:
            # append the children of the child node to the parent node
            children = [command_graph.nodes[child_id]
                        for child_id in command_graph.successors(node_b['id'])]
            for child in children:
                command_graph.add_edge(node_a['id'], child['id'])
                command_graph.remove_edge(node_b['id'], child['id'])
            # remove the child node,
            command_graph.remove_node(node_b['id'])
        elif match_type == "iu":
            children_a = [command_graph.nodes[child_id]
                          for child_id in command_graph.successors(node_a['id'])]
            children_a_not_b = [
                child for child in children_a if child['id'] != node_b['id']]
            children_b = [command_graph.nodes[child_id]
                          for child_id in command_graph.successors(node_b['id'])]

            # Convert the i into a u
            node_a['command']['symbol'] = "union"
            node_a['label'] = "union"
            new_id = max(list(command_graph.nodes)) + 1
            for child in children_a:
                command_graph.remove_edge(node_a['id'], child['id'])
            # for child in children_b:
            #     command_graph.remove_edge(node_b['id'], child['id'])
            # command_graph.remove_node(node_b['id'])

            for child in children_b:
                # create n new i nodes where n is the number of children of the u node
                command = {"type": "B", "symbol": "intersection"}
                command_graph.add_node(
                    new_id, label="intersection", command=command, id=new_id)
                for not_b_child in children_a_not_b:
                    command_graph.add_edge(new_id, not_b_child['id'])
                command_graph.add_edge(new_id, child['id'])
                command_graph.add_edge(node_a['id'], new_id)
                new_id += 1
                # where each node has all the children of the i node <not U> and one child of the u node

        return command_graph

    def extract_cmd_list(self, graph):
        # traverse the graph and extract the command list
        cmd_list = []

        node_list = [graph.nodes[1]]
        while node_list:
            cur_node = node_list.pop()
            node_id = cur_node['id']
            # Current Node op:
            if "command" in cur_node.keys():
                command = cur_node['command']
                cmd_list.append(command)
            children = [successor for successor in graph.successors(node_id)]
            children = [graph.nodes[child_id] for child_id in children]
            node_list.extend(children[::-1])

        return cmd_list

    def collapse_graph(self, command_graph, n_prims, draw_start):

        # RULES:
        # Intersection-> INtersection = collapse
        # Union -> union = collapse
        # Union -> Intersection = retain
        # Intersection -> Union = invert
        rule_match = None
        while (True):
            rule_match = self.get_rule_match(command_graph, only_simplify=True)
            if not rule_match:
                rule_match = self.get_rule_match(
                    command_graph, only_simplify=False)
                if not rule_match:
                    break
                else:
                    command_graph = self.resolve_rule(
                        command_graph, rule_match)
            else:
                command_graph = self.resolve_rule(command_graph, rule_match)

        # now convert the intersection nodes into intersection arrays:
        intersection_matrix = self.get_intersection_array(
            command_graph, n_prims, draw_start)

        return intersection_matrix, command_graph

    def get_intersection_array(self, command_graph, n_prims, draw_start):

        intersection_matrix = []
        node_list = [command_graph.nodes[1]]
        zero_row = np.zeros((n_prims), dtype=bool)
        while node_list:
            cur_node = node_list.pop()
            command = cur_node['command']
            c_type = command['type']
            if c_type == "B":
                node_id = cur_node['id']
                children = [command_graph.nodes[child_id]
                            for child_id in command_graph.successors(node_id)]
                node_list.extend(children[::-1])
                c_symbol = command['symbol']
                if c_symbol == "union":
                    # for each child if draw, create a row:
                    for child in children:
                        if child['command']['type'] == "D":
                            id = child['command']['ID']
                            loc_pointer = draw_start[child['command']
                                                     ['symbol']] + id
                            cur_row = zero_row.copy()  # np.zeros((n_prims), dtype=bool)
                            cur_row[loc_pointer] = True
                            intersection_matrix.append(cur_row)

                elif c_symbol == "intersection":
                    cur_row = zero_row.copy()
                    for child in children:
                        if child['command']['type'] == "D":
                            id = child['command']['ID']
                            loc_pointer = draw_start[child['command']
                                                     ['symbol']] + id
                            cur_row[loc_pointer] = True
                    intersection_matrix.append(cur_row)
        # intersection_matrix = np.array(intersection_matrix)
        intersection_matrix = np.array(intersection_matrix)
        # intersection_matrix = th.tensor(
        #     intersection_matrix, device=self.device, dtype=bool)
        intersection_matrix = th.tensor(
            intersection_matrix, dtype=bool)
        intersection_matrix = intersection_matrix.transpose(0, 1)
        return intersection_matrix

    def fast_compile(self, cmd_list, draw_count):
        # TODO: Improve this
        n_prim = 0
        # Draw cache size
        draw_start = dict()
        draw_transforms = dict()
        for draw_type in self.draw_seq:
            draw_start[draw_type] = n_prim
            n_prim += draw_count[draw_type]
            draw_transforms[draw_type] = th.zeros(
                (draw_count[draw_type], 3, 3), device=self.device)

        # Pass 1: Construct Draw Transforms, Remove Difference, Apply complements, And produce boolean formula.
        draw_transforms, inversion_array, no_complement_graph = self.reduce_cmd_list(
            cmd_list, draw_transforms, n_prim, draw_start)

        intersection_matrix, _ = self.collapse_graph(
            no_complement_graph, n_prim, draw_start)

        return draw_transforms, inversion_array, intersection_matrix

    def reduce_cmd_list(self, cmd_list, draw_transforms, n_prim, draw_start):

        # Transforms
        transform_stack = [self.transforms.get_affine_identity()]

        # Inversion
        inversion_array = np.zeros((n_prim), dtype=bool)
        inversion_mode = False
        inversion_stack = [inversion_mode]

        # Boolean Formula:
        no_complement_graph = nx.DiGraph()
        # no_complement_graph = nx.PyGraph()
        no_complement_graph.add_node(0, label="ROOT", id=0)
        graph_pointers = [0]
        new_cmd_list = []

        for ind, cmd in enumerate(cmd_list):
            c_type = cmd['type']
            c_symbol = cmd['symbol']
            inversion_mode = inversion_stack.pop()

            parent_pointer = graph_pointers.pop()
            child_pointer = len(no_complement_graph.nodes)
            if c_type == "B":
                latest_transform = transform_stack[-1]
                transform_stack.append(latest_transform)
                # add to graph:
                if c_symbol == "difference":
                    inversion_stack.append(not inversion_mode)
                else:
                    inversion_stack.append(inversion_mode)
                inversion_stack.append(inversion_mode)

                if inversion_mode:
                    current_symbol = INVERTED_MAP[c_symbol]
                else:
                    current_symbol = NORMAL_MAP[c_symbol]
                cmd['symbol'] = current_symbol
                new_cmd_list.append(cmd)
                graph_pointers.append(child_pointer)
                graph_pointers.append(child_pointer)
                add_node = True

            elif c_type == "T":
                latest_transform = transform_stack.pop()
                new_transform = self.transform_to_execute[c_symbol](
                    param=cmd['param'])
                updated_transform = th.matmul(latest_transform, new_transform)
                transform_stack.append(updated_transform)
                inversion_stack.append(inversion_mode)
                # skip append?
                graph_pointers.append(parent_pointer)
                add_node = False

            elif c_type == "D":
                # print("creating Draw", command)
                latest_transform = transform_stack.pop()
                # Parameters
                id = cmd['ID']
                draw_transforms[c_symbol][id] = latest_transform
                # Inversion notification:
                loc_pointer = draw_start[c_symbol] + id
                if inversion_mode:
                    inversion_array[loc_pointer] = True
                # add to new_cmd_list
                new_cmd_list.append(cmd)
                add_node = True
            if add_node:
                no_complement_graph.add_node(child_pointer,
                                             label=c_symbol,
                                             id=child_pointer,
                                             command=cmd)
                no_complement_graph.add_edge(parent_pointer, child_pointer)

        # inversion_array = th.from_numpy(inversion_array).to(self.device)
        inversion_array = th.from_numpy(inversion_array)
        inversion_array = inversion_array.unsqueeze(1)

        return draw_transforms, inversion_array, no_complement_graph

    def evaluate(self, draw_transforms, inversion_array, all_intersections, points=None):

        # load points
        if points is None:
            points = get_points(self.resolution)
        M = points.shape[0]
        # Add a fourth column of ones to the point cloud to make it homogeneous
        points_hom = th.cat([points, th.ones(M, 1)], dim=1).to(self.device)
        # First create all the primitives:
        all_primitives = []
        for draw_type, transforms in draw_transforms.items():
            # print(draw_type, transforms.shape)
            cur_points = points_hom.clone()
            # Apply the rotation matrices to the point cloud using einsum
            rotated_points_hom = th.einsum(
                'nij,mj->nmi', transforms, cur_points)
            # Extract the rotated points from the homogeneous coordinates
            rotated_points = rotated_points_hom[:, :, :2]

            draw_func = self.draw_to_execute[draw_type]
            primitives = draw_func(rotated_points)
            all_primitives.append(primitives)
        all_primitives = th.cat(all_primitives, dim=0)
        processed_primitives = th.where(
            inversion_array, -all_primitives, all_primitives)
        # Make this cheap:
        # P U N
        processed_primitives = processed_primitives.unsqueeze(1)
        processed_primitives = processed_primitives.expand(
            -1, all_intersections.shape[-1], -1)
        # P U N
        all_intersections = all_intersections.unsqueeze(-1)
        all_intersections = all_intersections.expand(
            -1, -1, processed_primitives.shape[2])
        fill = th.where(all_intersections, processed_primitives, self.neg_inf)
        # Intersections
        intersections = th.max(fill, 0)[0]
        # Unions
        output = th.min(intersections, 0)[0]

        return output

    def batch_evaluate_with_graph(self, collapsed_draws, batch_draw,
                                  collapsed_inversions, graphs, points=None):

        # create primitives:

        # first collapse all cuboids in one list
        if points is None:
            points = get_points(self.resolution)

        M = points.shape[0]
        # Add a fourth column of ones to the point cloud to make it homogeneous
        points_hom = th.cat([points, th.ones(M, 1)], dim=1).to(self.device)

        type_wise_primitives = dict()
        for draw_type, transforms in collapsed_draws.items():
            # print(draw_type, transforms.shape)
            if transforms == []:
                continue
            # transforms = th.stack(transforms, 0)
            cur_points = points_hom.clone()
            # Apply the rotation matrices to the point cloud using einsum
            transformed_points_hom = th.einsum(
                'nij,mj->nmi', transforms, cur_points)
            # Extract the rotated points from the homogeneous coordinates
            rotated_points = transformed_points_hom[:, :, :2]

            draw_func = self.draw_to_execute[draw_type]
            primitives = draw_func(rotated_points)
            # inversion = th.stack(collapsed_inversions[draw_type], 0).unsqueeze(1)
            inversion = collapsed_inversions[draw_type]
            sign_matrix = inversion * -2 + 1
            primitives = primitives * sign_matrix
            type_wise_primitives[draw_type] = primitives

        # next allot the primitive sequentially:
        # calculate offset for each graph
        type_wise_draw_count = defaultdict(int)
        all_outputs = []
        for ind, graph in enumerate(graphs):
            output = self.execute_graph(
                graph, type_wise_primitives, type_wise_draw_count)
            all_outputs.append(output)
            draw_specs = batch_draw[ind]
            for draw_type in type_wise_draw_count.keys():
                type_wise_draw_count[draw_type] += draw_specs[draw_type].shape[0]

        all_outputs = th.stack(all_outputs, 0)

        return all_outputs

    def execute_graph(self, cmd_list, primitive_dict, draw_count):

        canvas_stack = []
        # Given prefix notation - do postfix traversal in reverse order
        for cmd in cmd_list[::-1]:
            c_type = cmd['type']
            c_symbol = cmd['symbol']
            if c_type == "B":
                right_canvas = canvas_stack.pop()
                left_canvas = canvas_stack.pop()
                if c_symbol == "union":
                    new_canvas = th.minimum(left_canvas, right_canvas)
                else:
                    new_canvas = th.maximum(left_canvas, right_canvas)
                canvas_stack.append(new_canvas)
            else:
                # symbol = sp.symbols(c_symbol + "_" + str(cmd["ID"]))
                id = cmd['ID'] + draw_count[c_symbol]
                canvas = primitive_dict[c_symbol][id]
                canvas_stack.append(canvas)

        output = canvas_stack[0]
        # dnf_form = formula.to_dnf()
        return output

    def fast_sub_compile(self, cmd_list, draw_count):
        # TODO: Improve this
        n_prim = 0
        # Draw cache size
        draw_start = dict()
        draw_transforms = dict()
        draw_inversions = dict()
        for draw_type in self.draw_seq:
            draw_start[draw_type] = n_prim
            n_prim += draw_count[draw_type]
            draw_transforms[draw_type] = th.zeros(
                (draw_count[draw_type], 3, 3), device=self.device)
            # draw_inversions[draw_type] = th.zeros((draw_count[draw_type],), device=self.device, dtype=th.bool)
            # draw_transforms[draw_type] = np.zeros((draw_count[draw_type], 4, 4))
            draw_inversions[draw_type] = np.zeros(
                (draw_count[draw_type]), dtype=bool)

        # Pass 1: Construct Draw Transforms, Remove Difference, Apply complements, And produce boolean formula.
        draw_transforms, draw_inversions, no_complement_graph = self.reduce_cmd_list_diff(cmd_list,
                                                                                          draw_transforms, draw_inversions, n_prim, draw_start)

        # for draw_type in self.draw_seq:
        #     draw_inversions[draw_type] = th.from_numpy(draw_inversions[draw_type]).to(self.device)

        return draw_transforms, draw_inversions, no_complement_graph

    def reduce_cmd_list_diff(self, cmd_list, draw_transforms, draw_inversions, n_prim, draw_start):

        # Transforms
        transform_stack = [self.transforms.get_affine_identity()]

        # Inversion
        inversion_mode = False
        inversion_stack = [inversion_mode]

        # Boolean Formula:
        # no_complement_graph = nx.DiGraph()
        # no_complement_graph.add_node(0, label="ROOT", id=0)
        # graph_pointers = [0]

        new_cmd_list = []

        for ind, cmd in enumerate(cmd_list):
            c_type = cmd['type']
            c_symbol = cmd['symbol']
            inversion_mode = inversion_stack.pop()

            if c_type == "B":
                latest_transform = transform_stack[-1]
                transform_stack.append(latest_transform)
                # add to graph:
                if c_symbol == "difference":
                    inversion_stack.append(not inversion_mode)
                else:
                    inversion_stack.append(inversion_mode)
                inversion_stack.append(inversion_mode)

                if inversion_mode:
                    current_symbol = INVERTED_MAP[c_symbol]
                else:
                    current_symbol = NORMAL_MAP[c_symbol]
                cmd['symbol'] = current_symbol
                new_cmd_list.append(cmd)

            elif c_type == "T":
                latest_transform = transform_stack.pop()
                new_transform = self.transform_to_execute[c_symbol](
                    param=cmd['param'])
                updated_transform = th.matmul(new_transform, latest_transform)
                transform_stack.append(updated_transform)
                inversion_stack.append(inversion_mode)
                # skip append?

            elif c_type == "D":
                # print("creating Draw", command)
                latest_transform = transform_stack.pop()
                # Parameters
                id = cmd['ID']
                draw_transforms[c_symbol][id] = latest_transform
                # Inversion notification:
                # loc_pointer = draw_start[c_symbol] + id
                if inversion_mode:
                    draw_inversions[c_symbol][id] = True
                # add to new_cmd_list
                new_cmd_list.append(cmd)

        # inversion_array = th.from_numpy(inversion_array).to(self.device)
        # inversion_array = inversion_array.unsqueeze(1)

        return draw_transforms, draw_inversions, new_cmd_list
