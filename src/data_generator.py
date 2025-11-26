import xml.etree.ElementTree as ET
import networkx as nx
from typing import List, Set, Tuple, Dict
from os.path import join, exists
from argparse import ArgumentParser
import os
from tqdm import tqdm
from typing import cast
import dataclasses
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools

USE_CPU = cpu_count()


def extract_line_number(idx: int, nodes: List) -> int:
    """
    return the line number of node index

    Args:
        idx (int): node index
        nodes (List)
    Returns: line number of node idx
    """
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except Exception as e:
                    print(e)
                    pass
        idx -= 1
    return -1


def read_csv(csv_file_path: str) -> List:
    """
    read csv file
    """
    assert exists(csv_file_path), f"no {csv_file_path}"
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def extract_nodes_with_location_info(nodes):
    """
    Will return an array identifying the indices of those nodes in nodes array
    another array identifying the node_id of those nodes
    another array indicating the line numbers
    all 3 return arrays should have same length indicating 1-to-1 matching.
    
    """

    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number


def build_PDG(code_path: str, sensi_api_path: str,
              source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    nodes_path = join(code_path, "nodes.csv")
    edges_path = join(code_path, "edges.csv")
    
    print(f"Building PDG for: {code_path}")
    print(f"Checking nodes.csv: {exists(join(code_path, 'nodes.csv'))}")
    print(f"Checking edges.csv: {exists(join(code_path, 'edges.csv'))}")
    
    print(f"Looking for nodes.csv at: {nodes_path}")
    print(f"Looking for edges.csv at: {edges_path}")
    
    if not exists(nodes_path):
        print(f"Error: nodes.csv not found at {nodes_path}")
        return None, None
    if not exists(edges_path):
        print(f"Error: edges.csv not found at {edges_path}")
        return None, None
    """
    build program dependence graph from code

    Args:
        code_path (str): source code root path
        sensi_api_path (str): path to sensitive apis
        source_path (str): source file path

    Returns: (PDG, key line map)
    """
    nodes_path = join(code_path, "nodes.csv")
    edges_path = join(code_path, "edges.csv")
    assert exists(sensi_api_path), f"{sensi_api_path} not exists!"
    with open(sensi_api_path, "r", encoding="utf-8") as f:
        sensi_api_set = set([api.strip() for api in f.read().split(",")])
    if not exists(nodes_path) or not exists(edges_path):
        return None, None
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()
    if len(nodes) == 0:
        return None, None
    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name is None or function_name.strip() == '':
                continue
            if function_name.strip() in sensi_api_set:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    call_lines.add(line_no)
        elif ntype == 'ArrayIndexing':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                array_lines.add(line_no)
        elif ntype == 'PtrMemberAccess':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                ptr_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                arithmatic_lines.add(line_no)

    PDG = nx.DiGraph(file_paths=[source_path])
    control_edges, data_edges = list(), list()
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(
        nodes)
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_ln.keys(
            ) or end_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = node_id_to_ln[end_node_id]
            if edge_type == 'CONTROLS':  # Control
                control_edges.append((start_ln, end_ln, {"c/d": "c"}))
            if edge_type == 'REACHES':  # Data
                data_edges.append((start_ln, end_ln, {"c/d": "d"}))
    PDG.add_edges_from(control_edges)
    PDG.add_edges_from(data_edges)
    return PDG, {
        "call": call_lines,
        "array": array_lines,
        "ptr": ptr_lines,
        "arith": arithmatic_lines
    }


def build_XFG(PDG: nx.DiGraph, key_line_map: Dict[str, Set[int]],
              vul_lines: Set[int] = None) -> Dict[str, List[nx.DiGraph]]:
    """
    build XFGs
    Args:
        PDG (nx.DiGraph): program dependence graph
        key_line_map (Dict[str, Set[int]]): key lines
    Returns: XFG map
    """
    if PDG is None or key_line_map is None:
        return None
    # ct0, ct1 = 0, 0
    res = {"call": [], "array": [], "ptr": [], "arith": []}
    for key in ["call", "array", "ptr", "arith"]:
        for ln in key_line_map[key]:
            sliced_lines = set()

            # backward traversal
            bqueue = list()
            visited = set()
            bqueue.append(ln)
            visited.add(ln)
            while bqueue:
                fro = bqueue.pop(0)
                sliced_lines.add(fro)
                if fro in PDG._pred:
                    for pred in PDG._pred[fro]:
                        if pred not in visited:
                            visited.add(pred)
                            bqueue.append(pred)

            # forward traversal
            fqueue = list()
            visited = set()
            fqueue.append(ln)
            visited.add(ln)
            while fqueue:
                fro = fqueue.pop(0)
                sliced_lines.add(fro)
                if fro in PDG._succ:
                    for succ in PDG._succ[fro]:
                        if succ not in visited:
                            visited.add(succ)
                            fqueue.append(succ)
            if len(sliced_lines) != 0:
                XFG = PDG.subgraph(list(sliced_lines)).copy()
                XFG.graph["key_line"] = ln
                if vul_lines is not None:
                    if len(sliced_lines.intersection(vul_lines)) != 0:
                        XFG.graph["label"] = 1
                        # ct1 += 1
                    else:
                        XFG.graph["label"] = 0
                        # ct0 += 1
                else:
                    XFG.graph["label"] = "UNK"
                res[key].append(XFG)
        # print("ct1:", ct1)
        # print("ct0:", ct0)

    return res


def getCodeIDtoPathDict(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    '''build code testcaseid to path map

    use the manifest.xml. build {testcaseid:{filePath:set(vul lines)}}
    filePath use relevant path, e.g., CWE119/cve/source-code/project_commit/...
    :param testcases:
    :return: {testcaseid:{filePath:set(vul lines)}}
    '''
    codeIDtoPath: Dict[str, Dict[str, Set[int]]] = {}
    for testcase in testcases:
        files = testcase.findall("file")
        testcaseid = testcase.attrib["id"]
        codeIDtoPath[testcaseid] = dict()

        for file in files:
            path = file.attrib["path"]
            flaws = file.findall("flaw")
            mixeds = file.findall("mixed")
            fix = file.findall("fix")
            # print(mixeds)
            VulLine = set()
            if (flaws != [] or mixeds != [] or fix != []):
                # targetFilePath = path
                if (flaws != []):
                    for flaw in flaws:
                        VulLine.add(int(flaw.attrib["line"]))
                if (mixeds != []):
                    for mixed in mixeds:
                        VulLine.add(int(mixed.attrib["line"]))

            codeIDtoPath[testcaseid][path] = VulLine

    return codeIDtoPath


def dump_XFG(res: Dict[str, List[nx.DiGraph]], out_root_path: str,
             testcaseid: str):
    """
    dump XFG to file

    Args:
        res: XFGs
        out_root_path: output root path
        testcaseid: testcase id
    Returns:
    """
    if res is None:
        return
    testcase_out_root_path = join(out_root_path, testcaseid)
    if not exists(testcase_out_root_path):
        os.makedirs(testcase_out_root_path)
    for k in res:
        k_root_path = join(testcase_out_root_path, k)
        if not exists(k_root_path):
            os.makedirs(k_root_path)
        for XFG in res[k]:
            out_path = join(k_root_path, f"{XFG.graph['key_line']}.xfg.pkl")
            nx.write_gpickle(XFG, out_path)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


@dataclasses.dataclass
class QueueMessage:
    XFG_res: Dict
    out_root_path: str
    testcaseid: str
    is_finished: bool = False


def handle_queue_message(queue: Queue):
    """处理队列消息"""
    xfg_ct = 0
    processed_count = 0
    while True:
        try:
            message: QueueMessage = queue.get(timeout=300)  # 添加超时
            if message.is_finished:
                break
            if message.XFG_res is not None:
                dump_XFG(message.XFG_res, message.out_root_path, message.testcaseid)
                for k in message.XFG_res:
                    xfg_ct += len(message.XFG_res[k])
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} XFG groups, total {xfg_ct} XFGs")
        except Exception as e:
            print(f"Error in queue handler: {e}")
            break
    return xfg_ct


def process_parallel(testcase: ET.Element, queue: Queue, doneIDs: Set, codeIDtoPath: Dict, cwe_root: str,
                     source_root_path: str, out_root_path: str, sensi_api_path: str):
    """
    Args:
        testcase: XML testcase element
        queue: 消息队列
        doneIDs: 已完成的ID集合
        codeIDtoPath: 代码路径映射
        cwe_root: CWE根目录
        source_root_path: 源代码根路径
        out_root_path: 输出根路径
        sensi_api_path: 敏感API文件路径
    """
    testcaseid = testcase.attrib["id"]
    if testcaseid in doneIDs:
        print(f"Testcase {testcaseid} already processed, skipping...")
        return testcaseid
        
    if testcaseid in codeIDtoPath:
        file_map = codeIDtoPath[testcaseid]
        for file_path in file_map:
            print(f"Processing testcase {testcaseid}, file: {file_path}")
            
            vul_lines = file_map[file_path]
            
            # 修复路径构建
            # 假设CSV文件与源文件在相同目录结构下，但在不同的根目录中
            csv_path = join(cwe_root, "csv", file_path)
            
            # 检查CSV目录是否存在
            if not exists(csv_path):
                print(f"Warning: CSV directory does not exist: {csv_path}")
                continue
                
            source_path = join(source_root_path, file_path)
            
            print(f"CSV path: {csv_path}")
            print(f"Source path: {source_path}")
            print(f"Vulnerable lines: {vul_lines}")
            
            try:
                PDG, key_line_map = build_PDG(csv_path, sensi_api_path, source_path)
                
                if PDG is None or key_line_map is None:
                    print(f"Warning: Failed to build PDG for {file_path}")
                    continue
                    
                print(f"PDG built successfully with {len(PDG.nodes())} nodes and {len(PDG.edges())} edges")
                print(f"Key lines found: {key_line_map}")
                
                res = build_XFG(PDG, key_line_map, vul_lines)
                
                if res is not None:
                    queue.put(QueueMessage(res, out_root_path, testcaseid))
                    print(f"XFG generated for {file_path}")
                else:
                    print(f"Warning: Failed to build XFG for {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                
    return testcaseid
    """

    Args:
        testcase:
        doneIDs:
        codeIDtoPath:
        cwe_root:
        source_root_path:
        out_root_path:

    Returns:

    """
    testcaseid = testcase.attrib["id"]
    if testcaseid in doneIDs:
        return testcaseid
    if testcaseid in codeIDtoPath:
        file_map = codeIDtoPath[testcaseid]
        for file_path in file_map:
            # print(file_path)
            vul_lines = file_map[file_path]
            csv_path = join(cwe_root, "csv", file_path)
            source_path = join(source_root_path, file_path)
            PDG, key_line_map = build_PDG(csv_path, "data/sensiAPI.txt",
                                          source_path)
            res = build_XFG(PDG, key_line_map, vul_lines)
            queue.put(QueueMessage(res, out_root_path, testcaseid))
            # dump_XFG(res, out_root_path, testcaseid)
    return testcaseid


def check_environment(config_path: str):
    """检查环境和路径配置"""
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.data_folder
    cweid = config.dataset.name
    cwe_root = join(root, cweid)
    
    print("=== Environment Check ===")
    print(f"Data folder exists: {exists(root)}")
    print(f"CWE root exists: {exists(cwe_root)}")
    print(f"Source code exists: {exists(join(cwe_root, 'source-code'))}")
    print(f"CSV directory exists: {exists(join(cwe_root, 'csv'))}")
    print(f"Sensi API file exists: {exists('data/sensiAPI.txt')}")
    
    # 检查manifest.xml
    xml_path = join(cwe_root, "source-code", "manifest.xml")
    if exists(xml_path):
        tree = ET.ElementTree(file=xml_path)
        testcases = tree.findall("testcase")
        print(f"Found {len(testcases)} testcases in manifest.xml")
    else:
        print("manifest.xml not found!")

def generate(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.data_folder
    cweid = config.dataset.name
    cwe_root = join(root, cweid)
    source_root_path = join(cwe_root, "source-code")
    out_root_path = join(cwe_root, "XFG")
    xml_path = join(source_root_path, "manifest.xml")
    sensi_api_path = "data/sensiAPI.txt"

    print(f"CWE Root: {cwe_root}")
    print(f"Source Root Path: {source_root_path}")
    print(f"Output Path: {out_root_path}")
    print(f"Manifest XML: {xml_path}")
    print(f"Sensitive API Path: {sensi_api_path}")

    if not exists(xml_path):
        print(f"Error: manifest.xml not found at {xml_path}")
        return

    tree = ET.ElementTree(file=xml_path)
    testcases = tree.findall("testcase")
    codeIDtoPath = getCodeIDtoPathDict(testcases, source_root_path)

    print(f"Found {len(testcases)} testcases")

    if not exists(out_root_path):
        os.makedirs(out_root_path)
        
    done_file_path = join(cwe_root, "doneID.txt")
    if not exists(done_file_path):
        with open(done_file_path, "w", encoding="utf-8") as f:
            f.write("")
            
    with open(done_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        doneIDs = set(content.split(",")) if content else set()
        
    print(f"Already processed {len(doneIDs)} testcases")

    testcase_len = len(testcases)
    
    # 修复多进程池管理
    try:
        with Manager() as m:
            message_queue = m.Queue()
            
            # 使用更小的进程池，避免资源竞争
            num_processes = min(USE_CPU, 4)  # 限制最大进程数
            print(f"Using {num_processes} processes")
            
            with Pool(processes=num_processes) as pool:
                # 先启动队列处理器
                xfg_ct = pool.apply_async(handle_queue_message, (message_queue,))
                
                # 使用 partial 传递所有必要参数
                process_func = functools.partial(
                    process_parallel, 
                    queue=message_queue, 
                    doneIDs=doneIDs,
                    codeIDtoPath=codeIDtoPath,
                    cwe_root=cwe_root, 
                    source_root_path=source_root_path,
                    out_root_path=out_root_path,
                    sensi_api_path=sensi_api_path
                )
                
                testcaseids_done = []
                try:
                    for testcaseid in tqdm(
                        pool.imap_unordered(process_func, testcases),
                        desc="Processing testcases",
                        total=testcase_len,
                    ):
                        testcaseids_done.append(testcaseid)
                except Exception as e:
                    print(f"Error during processing: {e}")
                
                # 发送结束信号
                message_queue.put(QueueMessage(None, None, None, True))
                
                # 等待队列处理器完成
                pool.close()
                pool.join()
                
            xfg_count = xfg_ct.get()
            print(f"Total {xfg_count} XFGs generated!")
            
    except Exception as e:
        print(f"Error in multiprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    # 更新完成的任务ID
    new_done_ids = set(testcaseids_done) - doneIDs
    if new_done_ids:
        with open(done_file_path, 'a', encoding="utf-8") as f:
            for testcaseid in new_done_ids:
                f.write(str(testcaseid))
                f.write(",")
        print(f"Updated doneID.txt with {len(new_done_ids)} new testcases")


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    
    # 先检查环境
    check_environment(__args.config)
    
    # 然后运行生成过程
    generate(__args.config)