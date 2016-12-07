#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest

from aeneas.tree import Tree


class TestTree(unittest.TestCase):

    def create_tree1(self, soon=True):
        root = Tree(value="root")
        c1 = Tree(value="c1")
        c11 = Tree(value="c11")
        c111 = Tree(value="c111")
        c1111 = Tree(value="c1111")
        c1112 = Tree(value="c1112")
        c1113 = Tree(value="c1113")
        if soon:
            root.add_child(c1)
            c1.add_child(c11)
            c11.add_child(c111)
            c111.add_child(c1111)
            c111.add_child(c1112)
            c111.add_child(c1113)
        else:
            c111.add_child(c1111)
            c111.add_child(c1112)
            c111.add_child(c1113)
            c11.add_child(c111)
            c1.add_child(c11)
            root.add_child(c1)
        return (root, c1, c11, c111, c1111, c1112, c1113)

    def create_tree2(self):
        root = Tree(value="r")
        c1 = Tree(value="c1")
        c2 = Tree(value="c2")
        c3 = Tree(value="c3")
        c4 = Tree(value="c4")

        c11 = Tree(value="c11")
        c12 = Tree(value="c12")
        c13 = Tree(value="c13")

        c21 = Tree(value="c21")
        c22 = Tree(value="c22")
        c23 = Tree(value="c23")
        c24 = Tree(value="c24")
        c25 = Tree(value="c25")

        c231 = Tree(value="c231")
        c232 = Tree(value="c232")

        root.add_child(c1)
        root.add_child(c2)
        root.add_child(c3)
        root.add_child(c4)

        c1.add_child(c11)
        c1.add_child(c12)
        c1.add_child(c13)

        c2.add_child(c21)
        c2.add_child(c22)
        c2.add_child(c23)
        c2.add_child(c24)
        c2.add_child(c25)

        c23.add_child(c231)
        c23.add_child(c232)
        return (root, c1, c11, c12, c13, c2, c21, c22, c23, c231, c232, c24, c25, c3, c4)

    def test_empty(self):
        root = Tree()
        self.assertEqual(len(root), 0)
        self.assertEqual(root.level, 0)
        self.assertEqual(root.height, 1)
        self.assertIsNone(root.value)
        self.assertTrue(root.is_root)
        self.assertTrue(root.is_leaf)
        self.assertTrue(root.is_empty)
        self.assertTrue(root.is_pleasant)
        self.assertEqual(root.children, [])
        self.assertEqual(root.subtree, [root])
        self.assertEqual(root.leaves, [root])
        self.assertEqual(root.vleaves, [None])
        self.assertEqual(root.leaves_not_empty, [])
        self.assertEqual(root.vleaves_not_empty, [])

    def test_value(self):
        root = Tree(value="root")
        self.assertIsNotNone(root.value)
        self.assertFalse(root.is_empty)
        self.assertEqual(root.vleaves, ["root"])

    def test_parent(self):
        root = Tree(value="root")
        self.assertIsNone(root.parent)
        self.assertTrue(root.is_root)

    def test_set_parent(self):
        root = Tree(value="root")
        new_root = Tree(value="newroot")
        root.parent = new_root
        self.assertIsNotNone(root.parent)
        self.assertFalse(root.is_root)

    def test_str(self):
        root = Tree(value="root")
        s = root.__str__()
        self.assertIsNotNone(s)

    def test_unicode(self):
        root = Tree(value="root")
        s = root.__unicode__()
        self.assertIsNotNone(s)

    def test_add_child(self):
        root = Tree(value="root")
        child1 = Tree(value="child1")
        child2 = Tree(value="child2")
        root.add_child(child1)
        root.add_child(child2)
        self.assertEqual(len(root), 2)
        self.assertEqual(root.level, 0)
        self.assertEqual(root.height, 2)
        self.assertTrue(root.is_root)
        self.assertFalse(root.is_leaf)
        self.assertEqual(root.children, [child1, child2])
        self.assertEqual(root.vchildren, ["child1", "child2"])
        self.assertEqual(root.leaves, [child1, child2])
        self.assertEqual(root.vleaves, ["child1", "child2"])
        for node in [child1, child2]:
            self.assertEqual(len(node), 0)
            self.assertEqual(node.level, 1)
            self.assertTrue(node.is_leaf)
            self.assertFalse(node.is_root)
            self.assertEqual(node.children, [])
            self.assertEqual(node.vchildren, [])

    def test_add_child_soon(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1(soon=True)
        self.assertEqual(root.level, 0)
        self.assertEqual(c1.level, 1)
        self.assertEqual(c11.level, 2)
        self.assertEqual(c111.level, 3)
        self.assertEqual(c1111.level, 4)
        self.assertEqual(c1111.level, 4)
        self.assertEqual(c1111.level, 4)

    def test_add_child_late(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1(soon=False)
        self.assertEqual(root.level, 0)
        self.assertEqual(c1.level, 1)
        self.assertEqual(c11.level, 2)
        self.assertEqual(c111.level, 3)
        self.assertEqual(c1111.level, 4)
        self.assertEqual(c1111.level, 4)
        self.assertEqual(c1111.level, 4)

    def test_add_child_not_tree(self):
        root = Tree(value="root")
        with self.assertRaises(TypeError):
            root.add_child("bad child")

    def test_height(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1(soon=True)
        self.assertEqual(root.height, 5)
        self.assertEqual(c1.height, 4)
        self.assertEqual(c11.height, 3)
        self.assertEqual(c111.height, 2)
        self.assertEqual(c1111.height, 1)
        self.assertEqual(c1112.height, 1)
        self.assertEqual(c1113.height, 1)

    def test_levels(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        self.assertEqual(c1111.levels, [[c1111]])
        self.assertEqual(c1112.levels, [[c1112]])
        self.assertEqual(c1113.levels, [[c1113]])
        self.assertEqual(c111.levels, [[c111], [c1111, c1112, c1113]])
        self.assertEqual(c11.levels, [[c11], [c111], [c1111, c1112, c1113]])
        self.assertEqual(c1.levels, [[c1], [c11], [c111], [c1111, c1112, c1113]])
        self.assertEqual(root.levels, [[root], [c1], [c11], [c111], [c1111, c1112, c1113]])

    def test_vlevels(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        self.assertEqual(c1111.vlevels, [["c1111"]])
        self.assertEqual(c1112.vlevels, [["c1112"]])
        self.assertEqual(c1113.vlevels, [["c1113"]])
        self.assertEqual(c111.vlevels, [["c111"], ["c1111", "c1112", "c1113"]])
        self.assertEqual(c11.vlevels, [["c11"], ["c111"], ["c1111", "c1112", "c1113"]])
        self.assertEqual(c1.vlevels, [["c1"], ["c11"], ["c111"], ["c1111", "c1112", "c1113"]])
        self.assertEqual(root.vlevels, [["root"], ["c1"], ["c11"], ["c111"], ["c1111", "c1112", "c1113"]])

    def test_is_pleasant(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        self.assertTrue(root.is_pleasant)
        (root, c1, c11, c12, c13, c2, c21, c22, c23, c231, c232, c24, c25, c3, c4) = self.create_tree2()
        self.assertFalse(root.is_pleasant)

    def test_level_one_vs_children(self):
        (root, c1, c11, c12, c13, c2, c21, c22, c23, c231, c232, c24, c25, c3, c4) = self.create_tree2()
        for node in [root, c1, c23]:
            self.assertEqual(node.children, node.levels[1])
            self.assertEqual(node.children, node.level_at_index(1))
            self.assertEqual(node.vchildren, node.vlevels[1])
            self.assertEqual(node.vchildren, node.vlevel_at_index(1))

    def test_ancestor(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        self.assertEqual(root.ancestor(0), root)
        self.assertEqual(root.ancestor(1), None)
        self.assertEqual(root.ancestor(2), None)
        self.assertEqual(c1.ancestor(0), c1)
        self.assertEqual(c1.ancestor(1), root)
        self.assertEqual(c1.ancestor(2), None)
        self.assertEqual(c11.ancestor(0), c11)
        self.assertEqual(c11.ancestor(1), c1)
        self.assertEqual(c11.ancestor(2), root)
        self.assertEqual(c11.ancestor(3), None)
        self.assertEqual(c111.ancestor(0), c111)
        self.assertEqual(c111.ancestor(1), c11)
        self.assertEqual(c111.ancestor(2), c1)
        self.assertEqual(c111.ancestor(3), root)
        self.assertEqual(c111.ancestor(4), None)
        self.assertEqual(c1111.ancestor(0), c1111)
        self.assertEqual(c1111.ancestor(1), c111)
        self.assertEqual(c1111.ancestor(2), c11)
        self.assertEqual(c1111.ancestor(3), c1)
        self.assertEqual(c1111.ancestor(4), root)
        self.assertEqual(c1111.ancestor(5), None)

    def test_ancestor_bad(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        with self.assertRaises(TypeError):
            root.ancestor(None)
        with self.assertRaises(TypeError):
            root.ancestor(1.0)
        with self.assertRaises(ValueError):
            root.ancestor(-1)

    def keep(self, tree, levels):
        prev_levels = tree.levels
        if 0 in levels:
            s = 0
        else:
            s = 1
        for l in levels:
            s += len(prev_levels[l])
        tree.keep_levels(levels)
        self.assertEqual(len(tree.subtree), s)

    def test_keep_bad(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        with self.assertRaises(TypeError):
            root.keep_levels(None)
        with self.assertRaises(TypeError):
            root.keep_levels(1)
        with self.assertRaises(TypeError):
            root.keep_levels([1.0])

    def test_keep_levels_1(self):
        for levels in [
            [],
            [0],
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 4],
            [1, 3],
            [1, 3, 4],
            [1, 4],
            [2],
            [2, 3],
            [2, 3, 4],
            [2, 4],
            [3],
            [3, 4],
            [4]
        ]:
            (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
            self.keep(root, levels)

    def test_keep_levels_2(self):
        for levels in [
            [],
            [0],
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 3],
            [2],
            [2, 3],
            [3]
        ]:
            (root, c1, c11, c12, c13, c2, c21, c22, c23, c231, c232, c24, c25, c3, c4) = self.create_tree2()
            self.keep(root, levels)

    def test_clone(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        copy_root = root.clone()
        nodes = [n for n in root.dfs]
        copy_nodes = [n for n in copy_root.dfs]
        self.assertEqual(len(nodes), len(copy_nodes))
        for i in range(len(nodes)):
            self.assertEqual(nodes[i].value, copy_nodes[i].value)

    def test_clone_and_edit(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        copy_root = root.clone()
        copy_root.get_child(0).value = "n1"
        self.assertEqual(copy_root.get_child(0).value, "n1")
        self.assertEqual(root.get_child(0).value, "c1")

    def test_remove(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        self.assertEqual(len(c111.children), 3)
        c1113.remove()
        self.assertEqual(len(c111.children), 2)
        c1112.remove()
        self.assertEqual(len(c111.children), 1)
        c1111.remove()
        self.assertEqual(len(c111.children), 0)

    def test_remove_dangling(self):
        (root, c1, c11, c111, c1111, c1112, c1113) = self.create_tree1()
        self.assertEqual(len(list(root.dfs)), 7)
        self.assertEqual(len(list(c1.dfs)), 6)
        c1.remove()
        self.assertEqual(len(list(root.dfs)), 1)
        self.assertEqual(len(list(c1.dfs)), 6)


if __name__ == "__main__":
    unittest.main()
