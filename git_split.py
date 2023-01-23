from collections import defaultdict
from dataclasses import dataclass
from functools import cache, cached_property
import networkx as nx
import os
import re
from typing import Any, Dict, List, Optional, Set

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

def path_relative_to(path: str, relative_to: str) -> str:
    if path.startswith(relative_to):
        return path.removeprefix(relative_to).lstrip('/')
    raise ValueError(f'{path} is not a subpath of {relative_to}')

@dataclass
class PathData:
    path: str
    is_dir: bool

def path_data_from_tree(tree: Tree) -> List[PathData]:
    return [PathData(obj.abspath, isinstance(obj, Tree)) for obj in tree.traverse()]

def get_blob_contents(blob: Blob) -> bytes:
    return blob.data_stream[-1].read()

def module_refers_to_path(mod_name: str, path: str) -> bool:
    segs1 = [seg for seg in mod_name.split('.') if seg]
    segs2 = [seg for seg in path.removesuffix('__init__.py').removesuffix('.py').split('/') if seg]
    return segs1[:len(segs2)] == segs2

def get_best_ordered_partition(dg: nx.DiGraph) -> Partition:
    def _get_subpartition(subgraph: nx.DiGraph) -> Partition:
        # assume nodes are in a canonical ordering
        if (len(subgraph) == 0):
            return []
        assert sorted(list(subgraph)) == list(subgraph)
        n = subgraph.number_of_nodes()
        u = next(iter(subgraph))
        parents = set(subgraph.predecessors(u))
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
        if parents:
            for (i, part) in enumerate(subpartition):
                if parents.intersection(part):  # u joins the first component with a parent
                    part.insert(0, u)
                    break
        else:  # u gets its own component
            subpartition.insert(0, [u])
        assert (sum(map(len, subpartition)) == n)
        assert all(sorted(part) == part for part in subpartition)
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
    @property
    def repo_dir(self) -> str:
        assert (self.repo.working_tree_dir is not None)
        return str(self.repo.working_tree_dir)
    @cache
    def files_edited_by_commit(self, commit: Commit) -> List[str]:
        paths = []
        for path in commit.stats.files:
            match = RENAME_REGEX.match(path)
            if match:  # file was renamed
                groups = match.groups()
                ps = [groups[0] + groups[1], groups[0] + groups[2]]
            else:
                ps = [path]
            for p in ps:
                if os.path.isabs(p):
                    p = path_relative_to(p, self.repo_dir)
                paths.append(p)
        return paths
    @cached_property
    def all_file_paths(self) -> List[str]:
        all_paths: Set[str] = set()
        workdir = str(self.repo.working_tree_dir)
        for commit in self.history:
            all_paths.update([path_relative_to(pd.path, workdir) for pd in path_data_from_tree(commit.tree) if (not pd.is_dir)])
        return sorted(all_paths)
    @cached_property
    def all_python_file_paths(self) -> List[str]:
        return [path for path in self.all_file_paths if path.endswith('.py')]
    def python_dependencies(self, commit: Commit, path: str) -> List[str]:
        """For a given commit and file path, gets the set of files on which it depends (in the commit's tree)."""
        if path.endswith('.py'):
            dependencies = set()
            try:
                blob = commit.tree[path]
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
        """Gets a bipartite graph between commit indices and files."""
        g = nx.Graph()
        for (i, commit) in enumerate(self.history):
            for path in self.files_edited_by_commit(commit):
                g.add_edge(i, path)
        return g
    @cached_property
    def commit_file_dependency_graph(self) -> nx.DiGraph:
        """Dependency graph between (commit, file) pairs.
        A pair (C2, F2) depends on a pair (C1, F1) if F1 is a Python dependency of F2 (in C2), and C1 is an ancestor of C2."""
        dg = nx.DiGraph()
        commit_idx_by_path: Dict[str, List[int]] = defaultdict(list)
        for (i, commit) in enumerate(self.history):
            for path in self.files_edited_by_commit(commit):
                dg.add_node((i, path))
                deps = self.python_dependencies(commit, path)
                for dep in deps:
                    for j in commit_idx_by_path[dep]:
                        dg.add_edge((i, path), (j, dep))
                commit_idx_by_path[path].append(i)
        return dg
    @cached_property
    def commit_dependency_graph(self) -> nx.DiGraph:
        """Dependency graph on the chain of commits.
        Commit C2 depends on commit C1 (represented by an arrow from C2 to C1) if at least one file in C2 depends (in C2) on a file in C1."""
        dg1 = self.commit_file_dependency_graph
        dg2 = nx.DiGraph()
        paths_by_commit_idx = defaultdict(list)
        for (i, path) in dg1:
            paths_by_commit_idx[i].append(path)
            dg2.add_node(i)
        for (j, _) in enumerate(self.history):
            paths = paths_by_commit_idx[j]
            for p2 in paths:
                for (i, p1) in dg1[(j, p2)]:
                    # i must be an ancestor of j; p2 must depend on p1 (in commit j)
                    if (i != j):
                        dg2.add_edge(j, i)
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

