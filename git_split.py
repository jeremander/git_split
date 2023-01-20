from collections import defaultdict
from dataclasses import dataclass
from functools import cache, cached_property
import networkx as nx
import os
import re
from typing import Any, List, Optional, Set

from git import Blob, Commit, Repo, Tree


IMPORT_REGEX = re.compile(r'\s*from\s+([\w\.]+)\s+import.+|\s*import\s+([\w\.]+)\s*')
RENAME_REGEX = re.compile(r'([\w\.\/]*)\{([\w\.\/]+) => ([\w\.\/]+)\}')

Partition = List[List[Any]]

def get_import_from_line(line: str) -> Optional[str]:
    match = IMPORT_REGEX.match(line)
    if match:
        for s in match.groups():
            if s:
                return s
    return None

@dataclass
class PathData:
    path: str
    is_dir: bool

def path_data_from_tree(tree: Tree) -> List[PathData]:
    return [PathData(obj.abspath, isinstance(obj, Tree)) for obj in tree.traverse()]

@cache
def files_edited_by_commit(commit: Commit) -> List[str]:
    paths = []
    for path in commit.stats.files:
        match = RENAME_REGEX.match(path)
        if match:  # file was renamed
            groups = match.groups()
            ps = [groups[0] + groups[1], groups[0] + groups[2]]
        else:
            ps = [path]
        for p in ps:
            if (not os.path.isabs(p)):
                p = os.path.join(commit.repo.working_tree_dir, p)
            paths.append(p)
    return paths

def get_blob_contents(blob: Blob) -> bytes:
    return blob.data_stream[-1].read()

def module_refers_to_path(mod_name: str, path: str) -> bool:
    segs1 = [seg for seg in mod_name.split('.') if seg]
    segs2 = [seg for seg in path.removesuffix('.py').split('/') if seg]
    return segs2[-len(segs1):] == segs1

def get_best_ordered_partition(dg: nx.DiGraph) -> Partition:
    def _get_subpartition(subgraph: nx.DiGraph) -> Partition:
        # assume nodes are in a canonical ordering
        if (len(subgraph) == 0):
            return []
        # get the first parent of u (if it exists)
        n = subgraph.number_of_nodes()
        u = next(iter(subgraph))
        try:
            parent = next(subgraph.predecessors(u))
        except StopIteration:
            parent = None
        subgraph = subgraph.copy()
        # remove u and compute connected components
        subgraph.remove_node(u)
        component_sets = sorted(nx.connected_components(subgraph.to_undirected()), key = len)
        subpartition = []
        # recursively call this function on each subcomponent
        for comp_set in component_sets:
            comp = subgraph.subgraph(comp_set)
            subpartition += _get_subpartition(comp)
        # determine where to put u
        if (parent is None):  # u gets its own component
            subpartition.insert(0, [u])
        else:  # u joins the component of its parent
            for part in subpartition:
                if (parent in part):
                    part.insert(0, u)
        assert (sum(map(len, subpartition)) == n)
        return subpartition
    return _get_subpartition(dg)

def draw_graph_with_partition(dg, mode: str = 'spring') -> None:
    partition = get_best_ordered_partition(dg)
    idx_by_node = {}
    for (i, part) in enumerate(partition):
        for node in part:
            idx_by_node[node] = i
    node_color = [idx_by_node[node] for node in dg]
    labels = {node : i for (i, node) in enumerate(dg)}
    func = getattr(nx, f'draw_{mode}')
    func(dg, node_color = node_color, labels = labels, cmap = 'tab20')

@dataclass(frozen = True)
class GitSplitter:
    repo: Repo
    first_commit: Commit
    last_commit: Commit
    def __post_init__(self) -> None:
        assert self.repo.is_ancestor(self.first_commit, self.last_commit), f'commit {self.first_commit.hexsha} is not an ancestor of {self.last_commit.hexsha}'
    @property
    def history(self) -> List[Commit]:
        hist = []
        for commit in self.repo.iter_commits(self.last_commit):
            hist.append(commit)
            if (commit == self.first_commit):
                break
        return hist[::-1]
    @cached_property
    def all_file_paths(self) -> List[str]:
        all_paths: Set[str] = set()
        for commit in self.history:
            all_paths.update([pd.path for pd in path_data_from_tree(commit.tree) if (not pd.is_dir)])
        return sorted(all_paths)
    @cached_property
    def all_python_file_paths(self) -> List[str]:
        return [path for path in self.all_file_paths if path.endswith('.py')]
    def python_dependencies(self, commit: Commit, path: str) -> List[str]:
        """For a given commit and file path, gets the set of files on which it depends (in the commit's tree)."""
        if path.endswith('.py'):
            relpath = os.path.relpath(path, self.repo.working_tree_dir)
            dependencies = set()
            try:
                blob = commit.tree[relpath]
            except KeyError:
                return []
            for line in get_blob_contents(blob).split(b'\n'):
                if (b'import' in line):
                    mod = get_import_from_line(line.decode())
                    if mod:
                        for candidate in self.all_python_file_paths:
                            if module_refers_to_path(mod, candidate):
                                dependencies.add(candidate)
            return sorted(dependencies)
        return []
    @cached_property
    def commit_file_graph(self) -> nx.Graph:
        """Gets a bipartite graph between commits and files."""
        g = nx.Graph()
        for commit in self.history:
            for path in files_edited_by_commit(commit):
                g.add_edge(commit, path)
        return g
    @cached_property
    def commit_file_dependency_graph(self) -> nx.DiGraph:
        """Dependency graph between (commit, file) pairs.
        A pair (C2, F2) depends on a pair (C1, F1) if F1 is a Python dependency of F2 (in C2), and C1 is an ancestor of C2."""
        dg = nx.DiGraph()
        commits_by_path = defaultdict(list)
        for commit in self.history:
            for path in files_edited_by_commit(commit):
                dg.add_node((commit, path))
                deps = self.python_dependencies(commit, path)
                for dep in deps:
                    for ancestor in commits_by_path[dep]:
                        dg.add_edge((commit, path), (ancestor, dep))
                commits_by_path[path].append(commit)
        return dg
    @cached_property
    def commit_dependency_graph(self) -> nx.DiGraph:
        """Dependency graph on the chain of commits.
        Commit C2 depends on commit C1 (represented by an arrow from C2 to C1) if at least one file in C2 depends (in C2) on a file in C1."""
        dg1 = self.commit_file_dependency_graph
        dg2 = nx.DiGraph()
        nodes_by_commit = defaultdict(list)
        for (commit, path) in dg1:
            nodes_by_commit[commit].append((commit, path))
            dg2.add_node((commit, path))
        for commit in self.history:
            nodes = nodes_by_commit[commit]
            for (c2, p2) in nodes:
                for (c1, p1) in dg1[(c2, p2)]:
                    # c1 must be an ancestor of c2; p2 must depend on p1 (in c2)
                    if (c2 != c1):
                        dg2.add_edge(c2, c1)
        return dg2
    def best_commit_partition(self) -> Partition:
        return get_best_ordered_partition(self.commit_dependency_graph)
    def draw_commit_graph(self, mode: str = 'spring') -> None:
        dg = self.commit_dependency_graph
        draw_graph_with_partition(dg, mode = mode)


# tests

def make_graph(n, edges):
    dg = nx.DiGraph()
    dg.add_nodes_from(range(n))
    dg.add_edges_from(edges)
    return dg

dg1 = make_graph(3, [(2, 1)])
dg2 = make_graph(4, [(1, 0), (3, 0), (3, 2)])
dg3 = make_graph(5, [(1, 0), (2, 0), (3, 1), (4, 3)])
dg4 = make_graph(5, [(1, 0), (2, 0), (3, 1), (4, 2)])

