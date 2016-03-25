#!/usr/bin/env python
# coding=utf-8

"""
A generic rooted, ordered, levelled tree.

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Tree(object):
    """
    A generic rooted, ordered, levelled tree.

    Nodes can contain arbitrary (possibly, different) types of data.

    The children of a node are stored in the same order
    as they are inserted.

    The node keeps some extra information (parent node, level)
    which is useful when building or visiting a full tree.

    Two visits are implemented: depth-first and level-order,
    with the possibility of returning only the leaves.

    This class is optimized for ease of use
    with :class:`aeneas.textfile.TextFragment` and
    :class:`aeneas.syncmap.SyncMapFragment` objects,
    not for best performance or minimum memory footprint.
    Using this class should be fine for representing
    any reasonable text tree.

    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"Tree"

    def __init__(self, value=None, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
        self.value = value
        self.__children = []
        self.__parent = None
        self.__level = 0

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def __unicode__(self):
        return u"%s (l: %s, c: %s)" % (self.value, gf.safe_int(self.level), gf.safe_int(len(self)))

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def __len__(self):
        return len(self.children)

    @property
    def value(self):
        """
        The value stored in this node.

        :rtype: variant
        """
        return self.__value
    @value.setter
    def value(self, value):
        """
        Set the value stored in this node.

        :param variant value: the value/object to be stored
        """
        self.__value = value

    @property
    def children(self):
        """
        Return the list of the direct children of this node.

        :rtype: list of :class:`aeneas.tree.Tree`
        """
        return self.__children

    @property
    def vchildren(self):
        """
        Return the list of values of the direct children of this node.

        :rtype: list of variant 
        """
        return [n.value for n in self.children]

    @property
    def children_not_empty(self):
        """
        Return the list of the not empty direct children of this node.

        :rtype: list of :class:`aeneas.tree.Tree`
        """
        return [n for n in self.children if not n.is_empty]

    @property
    def vchildren_not_empty(self):
        """
        Return the list of values of the not empty direct children of this node.

        :rtype: list of :class:`aeneas.tree.Tree`
        """
        return [n.value for n in self.children_not_empty]

    @property
    def is_leaf(self):
        """
        Return ``True`` if this node is a leaf node.

        :rtype: bool
        """
        return len(self.children) == 0

    @property
    def is_empty(self):
        """
        Return ``True`` if this node is empty, i.e., it has no value.

        :rtype: bool
        """
        return self.value is None

    @property
    def parent(self):
        """
        Return the parent node of this node, or ``None`` if this node is a root.

        :rtype: :class:`aeneas.tree.Tree`
        """
        return self.__parent
    @parent.setter
    def parent(self, parent):
        """
        Set the parent of this node.

        :param parent: the parent node
        :type  parent: :class:`aeneas.tree.Tree`
        """
        self.__parent = parent

    @property
    def is_root(self):
        """
        Return ``True`` if this node is the root node.

        :rtype: bool
        """
        return self.__parent is None

    @property
    def level(self):
        """
        Return the level of this node,
        starting from ``0`` for the root,
        ``1`` for the direct children of the root,
        and so on.
        
        :rtype: int
        """
        return self.__level

    @property
    def is_pleasant(self):
        """
        Return ``True`` if all the leaves
        in the subtree rooted at this node
        are at the same level.

        :rtype: bool
        """
        levels = sorted([n.level for n in self.leaves])
        return levels[0] == levels[-1] 

    def add_child(self, node, as_last=True):
        """
        Add the given child to the current list of children.

        The new child is appended as the last child if ``as_last``
        is ``True``, or as the first child if ``as_last`` is ``False``.

        This call updates the ``__parent`` and ``__level`` fields of ``node``.

        :param node: the child node to be added
        :type  node: :class:`aeneas.tree.Tree`
        :param bool as_last: if ``True``, append the node as the last child;
                             if ``False``, append the node as the first child
        :raises: TypeError if ``node`` is not an instance of :class:`aeneas.tree.Tree`
        """
        if not isinstance(node, Tree):
            raise TypeError("node must be an instance of Tree")
        if as_last:
            self.__children.append(node)
        else:
            self.__children = [node] + self.__children
        node.__parent = self
        new_height = 1 + self.level
        for n in node.subtree:
            n.__level += new_height

    def remove_child(self, index):
        """
        Remove the child at the given index
        from the current list of children.

        :param int index: the index of the child to be removed
        """
        if index < 0:
            index = index + len(self)
        self.__children = self.__children[0:index] + self.__children[index+1:]

    def remove_children(self, reset_parent=True):
        """
        Remove all the children of this node. 

        :param bool reset_parent: if ``True``, set to ``None`` the parent attribute
                                  of the children
        """
        if reset_parent:
            for child in self.children:
                child.parent = None
        self.__children = []

    def get_child(self, index):
        """
        Return the child at the given index
        in the current list of children.

        :param int index: the index of the child to be returned
        """
        return self.children[index]

    def get_vchild(self, index):
        """
        Return the value of the child at the given index
        in the current list of children.

        :param int index: the index of the child to be returned
        """
        return self.get_child(index).value

    @property
    def subtree(self):
        """
        Return the list of the nodes in the tree rooted at this node, in DFS order.

        Note that this node is always the first element of the returned list.
        If you want to exclude it, use ``node.subtree[1:]``.

        :rtype: list of Tree
        """
        return list(self.dfs)

    @property
    def leaves(self):
        """
        Return the list of leaves
        in the tree rooted at this node,
        in DFS order.

        :rtype: list of :class:`aeneas.tree.Tree`
        """
        return [n for n in self.dfs if n.is_leaf]

    @property
    def vleaves(self):
        """
        Return the list of leaf values
        in the tree rooted at this node,
        in DFS order.

        :rtype: list of variant
        """
        return [n.value for n in self.leaves]

    @property
    def leaves_not_empty(self):
        """
        Return the list of leaves not empty
        in the tree rooted at this node,
        in DFS order.

        :rtype: list of :class:`aeneas.tree.Tree`
        """
        return [n for n in self.dfs if ((n.is_leaf) and (not n.is_empty))]

    @property
    def vleaves_not_empty(self):
        """
        Return the list of not empty leaf values
        in the tree rooted at this node,
        in DFS order.

        :rtype: list of variant
        """
        return [n.value for n in self.leaves_not_empty]

    @property
    def height(self):
        """
        Return the height of the tree
        rooted at this node,
        that is, the difference between the level
        of a deepest leaf and the level of this node.
        Return ``1`` for a single-node tree,
        ``2`` for a two-levels tree, etc.

        :rtype: int
        """
        return max([n.level for n in self.subtree]) - self.level + 1

    @property
    def dfs(self):
        """
        Depth-first search of the tree rooted at this node.
        (First visit children, then visit current node.)

        :rtype: generator of :class:`aeneas.tree.Tree`
        """
        for node in self.children:
            for v in node.dfs:
                yield v
        yield self

    @property
    def pre(self):
        """
        Pre-order search of the tree rooted at this node.
        (First visit current node, then visit children.)

        :rtype: generator of :class:`aeneas.tree.Tree`
        """
        yield self
        for node in self.children:
            for v in node.pre:
                yield v

    @property
    def levels(self):
        """
        Return a list of lists of nodes.
        The outer list is indexed by the level.
        Each inner list contains the nodes at that level,
        in DFS order.

        :rtype: list of lists of :class:`aeneas.tree.Tree`
        """
        ret = [[] for i in range(self.height)]
        for node in self.subtree:
            ret[node.level - self.level].append(node)
        return ret

    @property
    def vlevels(self):
        """
        Return a list of lists of node values.
        The outer list is indexed by the level.
        Each inner list contains the values of the nodes at that level,
        in DFS order.

        Note that values might be ``None``.

        :rtype: list of lists of variant
        """
        return [[n.value for n in level] for level in self.levels]

    def level_at_index(self, index):
        """
        Return the list of nodes at level ``index``,
        in DFS order.

        :param int index: the index
        :rtype: list of :class:`aeneas.tree.Tree`

        :raises: ValueError if the given ``index`` is not valid
        """
        levels = self.levels
        if (index < 0) or (index >= len(levels)):
            raise ValueError("The given level index is not valid")
        return self.levels[index]

    def vlevel_at_index(self, index):
        """
        Return the list of node values at level ``index``,
        in DFS order.

        :param int index: the index
        :rtype: list of :class:`aeneas.tree.Tree`
        
        :raises: ValueError if the given ``index`` is not valid
        """
        return [n.value for n in self.level_at_index(index)]

    def ancestor(self, index):
        """
        Return the ``index``-th ancestor.

        The 0-th ancestor is the node itself,
        the 1-th ancestor is its parent node,
        etc.

        :param int index: the number of levels to go up
        :rtype: :class:`aeneas.tree.Tree`
        :raises: TypeError if ``index`` is not an int
        :raises: ValueError if ``index`` is negative
        """
        if not isinstance(index, int):
            raise TypeError("index must be an int")
        if index < 0:
            raise ValueError("index cannot be negative")
        parent_node = self
        for i in range(index):
            if parent_node is None:
                break
            parent_node = parent_node.parent
        return parent_node

    def keep_levels(self, level_indices):
        """
        Rearrange the tree rooted at this node
        to keep only the given levels.

        The returned Tree will still be rooted
        at the current node, i.e. this function
        implicitly adds ``0`` to ``level_indices``.

        If ``level_indices`` is an empty list,
        only this node will be returned, with no children.

        Elements of ``level_indices`` that do not
        represent valid level indices (e.g., negative, or too large)
        will be ignored and no error will be raised.

        Important: this function modifies
        the original tree in place!

        :param list level_indices: the list of int, representing the levels to keep
        :raises: TypeError if ``level_indices`` is not a list or if
                 it contains an element which is not an int
        """
        if not isinstance(level_indices, list):
            raise TypeError("level_indices must be a list")
        for l in level_indices:
            if not isinstance(l, int):
                raise TypeError("The elements of level_indices must be int")
        prev_levels = self.levels
        level_indices = set(level_indices)
        if not 0 in level_indices:
            level_indices.add(0)
        level_indices = level_indices & set(range(self.height))
        level_indices = sorted(level_indices)[::-1]
        # first, remove children
        for l in level_indices:
            for node in prev_levels[l]:
                node.remove_children(reset_parent=False)
        # then, connect to the right new parent
        for i in range(len(level_indices) - 1):
            l = level_indices[i]
            for node in prev_levels[l]:
                parent_node = node.ancestor(l - level_indices[i+1])
                parent_node.add_child(node)



