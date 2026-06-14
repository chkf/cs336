/**
 * ===========================================================================
 * tokenizer_trainer.cc — BPE（Byte Pair Encoding）分词器训练器的 C++ 实现
 * ===========================================================================
 *
 * 本文件实现了一个高效的 BPE 分词器训练器，通过 pybind11 导出为 Python 模块。
 *
 * BPE 算法核心思想：
 *   1. 将所有文本拆成单个字符（如 "hello" → ["h","e","l","l","o"]）
 *   2. 统计所有相邻字符对的出现频率
 *   3. 合并频率最高的字符对，形成新 token（如 "l"+"l" → "ll"）
 *   4. 重复 2-3 步，直到达到目标词表大小
 *
 * 此文件包含了以下 C++ 语法知识点（按出现顺序）：
 *   - #include：头文件引入
 *   - using 类型别名
 *   - struct 结构体
 *   - class 类（构造/析构、public/private 访问控制）
 *   - template 模板
 *   - 位运算
 *   - static_cast 静态类型转换
 *   - const 常量修饰符
 *   - 引用 (&)
 *   - 匿名 namespace（内部链接）
 *   - Boost.MultiIndex 多索引容器
 *   - lambda 表达式
 *   - 初始化列表
 *   - std::vector / std::unordered_map 等 STL 容器
 *   - pybind11 绑定
 *   - 移动语义 (std::move)
 */

// ===========================================================================
// 头文件引入（#include）
// ===========================================================================
// #include 是 C++ 的预处理器指令，用于在编译前将指定头文件的内容"复制粘贴"到当前位置。
// <> 尖括号用于标准库或第三方库路径，编译器会在系统 include 目录中查找。
// 头文件通常包含类、函数的声明，让当前文件知道它们"长什么样"。

#include <pybind11/pybind11.h>   // pybind11 主头文件：将 C++ 代码绑定到 Python
#include <pybind11/stl.h>         // pybind11 STL 支持：自动转换 C++ 容器与 Python 容器（如 vector?list）
#include <boost/multi_index/hashed_index.hpp>    // Boost 多索引：哈希索引（O(1) 快速查找）
#include <boost/multi_index/member.hpp>           // Boost 多索引：按结构体成员字段建立索引
#include <boost/multi_index/ordered_index.hpp>    // Boost 多索引：有序索引（自动排序）
#include <boost/multi_index_container.hpp>        // Boost 多索引：多索引容器主模板
#include <cstdint>   // 固定宽度整数类型：uint32_t、uint64_t 等（跨平台一致，不像 int 大小可能不同）
#include <string>    // std::string：C++ 标准字符串类
#include <tuple>     // std::tuple：元组，可以存放多个不同类型值（类似 Python tuple）
#include <unordered_map>  // std::unordered_map：哈希表实现的映射容器（key→value，O(1) 平均查找）
#include <vector>    // std::vector：动态数组（类似 Python 的 list，但元素类型必须相同）

// ===========================================================================
// 命名空间别名
// ===========================================================================
// namespace 是 C++ 的命名空间，用于避免名称冲突（类似 Python 的模块名做前缀）。
// "py::" 是 "pybind11::" 的简写，方便后续代码中使用 py::class_ 等 pybind11 的功能。
namespace py = pybind11;

// ===========================================================================
// 类型别名（using）
// ===========================================================================
// using X = Y; 是 C++11 引入的类型别名语法（等价于传统的 typedef Y X;）。
// 它给一个已存在的类型起一个更可读的名字。
//
// 这里：
//   - Pair  = uint64_t：用 64 位无符号整数存储一对 token 索引
//   - Index = uint32_t：用 32 位无符号整数存储单个 token 在词表中的索引
//
// 为什么 Pair 用 64 位？
//   因为把两个 32 位的 Index 编码到一个 64 位整数中，可以高效比较、哈希、查找（见 MakePair）。
using Pair = uint64_t;
using Index = uint32_t;

// ===========================================================================
// 结构体（struct）—— PreTokenState
// ===========================================================================
// struct 是 C++ 中定义数据结构的关键字，类似 Python 的 dataclass。
// struct 默认所有成员是 public（公开的），而 class 默认是 private。
// 这里 struct 很适合纯数据容器。
struct PreTokenState {
    std::vector<Index> pre_token_index;  // 存储某个 pre-token 的 token 索引序列，如 [104, 101, ...]
    size_t count;                        // 该 pre-token 在语料中出现的次数（用于加权频次统计）
};

// ===========================================================================
// Pair 的编解码函数
// ===========================================================================

/**
 * MakePair：将两个 32 位索引打包成一个 64 位整数（Pair）
 *
 * 位运算详解：
 *   Pair 是 64 位（8 字节），Index 是 32 位（4 字节）。
 *   我们让高 32 位存储 first，低 32 位存储 second：
 *
 *   64-bit Pair:  |-------- first (32bit) --------|-------- second (32bit) --------|
 *                 ^                                 ^
 *                 高 32 位                           低 32 位
 *
 *   具体操作：
 *     static_cast<Pair>(first)  ← 首先把 32 位 first 提升为 64 位
 *     << 32                      ← 左移 32 位，把 first 放到高 32 位位置
 *     | second                   ← 按位或，把 second 放到低 32 位
 *
 *   static_cast<T>(x) 是 C++ 的静态类型转换，在编译时检查类型兼容性，
 *   比 C 风格的 (T)x 更安全（编译时会报类型不匹配的错误）。
 */
Pair MakePair(Index first, Index second) {
    return (static_cast<Pair>(first) << 32) | second;
}

/**
 * GetFirst：从 Pair 中提取 first（高 32 位）
 *
 *   pair >> 32  ← 右移 32 位，高 32 位信息被移到低 32 位，原低 32 位被丢弃
 *   static_cast<Index>(...) ← 再把 64 位截断回 32 位
 */
Index GetFirst(Pair pair) { return static_cast<Index>(pair >> 32); }

/**
 * GetSecond：从 Pair 中提取 second（低 32 位）
 *
 *   直接 static_cast<Index>(pair) 会把 64 位截断，只保留低 32 位（高位自动丢弃）。
 *   不需要做额外的位操作。
 */
Index GetSecond(Pair pair) { return static_cast<Index>(pair); }

// ===========================================================================
// 函数对象（Functor）—— PairCountCompare
// ===========================================================================
// struct 在 C++ 中不仅可以存放数据，还可以重载 operator() 使其成为一个"可调用对象"。
// 这种对象叫 Functor（函数对象），可以作为比较函数传给容器（如 std::sort 或 Boost 的 ordered_index）。
//
// 此处 PairCountCompare 用于在 Boost 多索引容器的有序索引中，定义 (Pair, 计数) 元素的排序规则。
// 排序优先级：1. 计数降序（高频优先） 2. 按 first token 的字符串比较 3. 按 second token 的字符串比较
struct PairCountCompare {
    /**
     * 构造函数 —— 初始化列表（Initializer List）
     *
     * : vocab(vocab)  这叫做初始化列表（member initializer list），在构造函数体 {} 之前执行。
     * 等价于在函数体里写 this->vocab = vocab;，但更高效（直接初始化而非先默认构造再赋值）。
     *
     * 参数中 const std::vector<std::string>& vocab：
     *   - const 表示这个引用指向的内容不可修改
     *   - & 表示引用（reference），不会拷贝整个 vector，传递的是原始对象的"别名"
     *
     * 成员变量是 const std::vector<std::string>& vocab;（引用类型成员），
     * 它引用外部的词表 vector，而不是复制一份（避免大量字符串拷贝）。
     */
    PairCountCompare(const std::vector<std::string>& vocab) : vocab(vocab) {}

    /**
     * string_greater：自定义字符串比较（相当于 > 操作符）
     *
     * const 后缀表示这是一个 const 成员函数，不会修改对象的成员变量。
     * 参数是 const std::string&：常量引用，避免拷贝字符串。
     *
     * static_cast<unsigned char>(s[i])：
     *   char 可能是 signed 或 unsigned（取决于编译器/平台）。
     *   转为 unsigned char 确保按字节值（0-255）比较，而非按有符号值（-128~127）比较。
     *   否则某些字符（ASCII > 127）会变成负数，导致排序混乱。
     */
    bool string_greater(const std::string& s1, const std::string& s2) const {
        // std::min(a, b) 返回较小值，取两个字符串的较短长度
        size_t len = std::min(s1.size(), s2.size());
        for (size_t i = 0; i < len; ++i) {  // ++i 前置自增（比 i++ 稍高效，这里区别不大）
            if (s1[i] != s2[i]) {
                return static_cast<unsigned char>(s1[i]) >
                       static_cast<unsigned char>(s2[i]);
            }
        }
        // 所有字符都相同，则较长的字符串"更大"
        return s1.size() > s2.size();
    }

    /**
     * operator() 重载 —— 让这个 struct 变成可调用对象
     *
     * 定义了两个参数的 operator()，当比较两个 (Pair, size_t) 元素时，
     * 容器会自动调用这个函数来决定排序顺序。
     *
     * const std::pair<Pair, size_t>& a：
     *   std::pair 是 STL 的工具类，存放两个值（类似 Python 二元 tuple）。
     *   a.first 是 Pair，a.second 是 size_t（即计数 count）。
     *
     * 排序规则：先按 count 降序，count 相同则按 token 字符串比较
     */
    bool operator()(const std::pair<Pair, size_t>& a,
                    const std::pair<Pair, size_t>& b) const {
        // 1. 先比较计数（频次），高频优先
        if (a.second != b.second) {
            return a.second > b.second;  // 降序：计数大的排在前面
        }

        // 2. 计数相同，比较 first token 的字符串
        const std::string& a_first = vocab[GetFirst(a.first)];
        const std::string& b_first = vocab[GetFirst(b.first)];

        if (a_first != b_first) {
            return string_greater(a_first, b_first);
        }

        // 3. first 也相同，比较 second token 的字符串
        const std::string& a_second = vocab[GetSecond(a.first)];
        const std::string& b_second = vocab[GetSecond(b.first)];

        return string_greater(a_second, b_second);
    }

   private:
    // 引用成员：不拷贝词表的副本，而是引用外部的词表
    // 使用引用作为成员需要特别注意：引用的对象必须在本对象存活期间一直存在
    const std::vector<std::string>& vocab;
};

// ===========================================================================
// 匿名 namespace（Anonymous Namespace）
// ===========================================================================
// namespace { ... } 是匿名命名空间。
// 在匿名命名空间内定义的内容有**内部链接（internal linkage）**，
// 仅在当前 .cc 文件内可见，外部文件无法访问（类似 C 中的 static 全局变量/函数）。
// 这是现代 C++ 推荐的做法，用于替代 static 全局声明。
//
// using namespace boost::multi_index; 表示在当前作用域内，
// 可以直接使用 boost::multi_index 命名空间内的名字（如 multi_index_container），
// 而无需加 boost::multi_index:: 前缀。
namespace {
using namespace boost::multi_index;

// ---------------------------------------------------------------------------
// Boost.MultiIndex 多索引容器 —— PairCountContainer
// ---------------------------------------------------------------------------
// 这是一个模板 typedef（typedef 用 multi_index_container<...> 定义了一个新类型）。
//
// multi_index_container 是 Boost 提供的一个功能强大的容器，它允许为同一份数据
// 同时维护多个索引。这里存储的元素类型是 std::pair<Pair, size_t>，即 (token对, 出现次数)。
//
// indexed_by<...> 定义了两个索引：
//
//   索引1 (hashed_unique)：
//     - hashed_unique：基于哈希的索引，不允许重复键（类似 std::unordered_set 的键）
//     - member<...>：指定使用元素 std::pair<Pair, size_t> 的 first 成员（即 Pair）作为键
//     - 用途：通过 Pair 快速查找/更新某个 token 对的计数（O(1) 均摊复杂度）
//
//   索引2 (ordered_non_unique)：
//     - ordered_non_unique：有序索引，允许重复键（类似 std::multiset）
//     - identity<...>：键就是元素本身（整个 pair）
//     - PairCountCompare：用我们自定义的比较器来决定排序顺序
//     - tag<struct by_count>：给这个索引起一个标签名 "by_count"，方便代码中引用
//     - 用途：快速找到当前计数最高的 token 对（因为按计数降序排列）
//
// 模板参数解释：
//   - multi_index_container<元素类型, indexed_by<索引定义...>>
//   - member<元素类型, 成员类型, 成员指针>  其中 &std::pair<Pair,size_t>::first 是指向成员的指针
//   - tag<...> 给索引一个"标签类型"（空 struct），通过标签类型而不是序号来访问特定索引
//
typedef multi_index_container<
    std::pair<Pair, size_t>,  // 存储的元素：token对 + 计数
    indexed_by<
        // 索引1：按 Pair 哈希，快速查找（O(1)）
        hashed_unique<member<std::pair<Pair, size_t>,
                             Pair,
                             &std::pair<Pair, size_t>::first>>,

        // 索引2：按计数排序（使用 PairCountCompare），找最大计数的 Pair
        ordered_non_unique<tag<struct by_count>,
                           identity<std::pair<Pair, size_t>>,
                           PairCountCompare>>>
    PairCountContainer;

}  // namespace

// ===========================================================================
// 类（class）—— TokenizerTrainerC
// ===========================================================================
// class 是 C++ 面向对象的核心语法。
// 与 struct 不同，class 的成员默认是 private（私有的）。
//
// public: 和 private: 是访问控制修饰符：
//   - public：外部可访问（类的接口）
//   - private：仅类内部可访问（实现细节，封装）
//
class TokenizerTrainerC {
   public:
    /**
     * 构造函数（Constructor）
     *
     * 当创建 TokenizerTrainerC 对象时自动调用。
     *
     * 初始化列表（: 后面的部分）：
     *   : target_vocab_size(target_vocab_size), pair_to_count(boost::make_tuple(...))
     *
     *   这是 C++ 初始化列表语法。对于成员变量，在初始化列表中初始化比在函数体 {} 里赋值更高效：
     *   初始化列表是"直接构造"，函数体赋值是"先默认构造再赋值"（多了临时对象的开销）。
     *
     *   pair_to_count 的初始化使用了 boost::make_tuple（类似 std::make_tuple），
     *   因为 Boost 多索引容器的构造函数接受两个索引的参数（用 tuple 打包传入）。
     *   参数中：
     *     0 表示哈希桶数设为 0（让 Boost 自动选择合适的桶数）
     *     boost::multi_index::member<...>() 指定成员指针作为键提取器
     *     boost::hash<Pair>() 是哈希函数对象
     *     std::equal_to<Pair>() 是相等比较函数对象
     *     boost::multi_index::identity<...>() 表示键就是元素本身
     *
     * for 循环体中：
     *   AddToken(std::string(1, static_cast<char>(i)))：
     *     std::string(1, c) 构造一个长度为1、内容为字符 c 的 string。
     *     static_cast<char>(i) 把 0-255 的整数转为 char 字符。
     *   BPE 的初始词表必须包含所有 256 个单字节（0x00-0xFF），
     *   因为任何文本都可以按字节拆分，保证所有输入都能被 tokenize。
     */
    TokenizerTrainerC(size_t target_vocab_size)
        : target_vocab_size(target_vocab_size),
          pair_to_count(boost::make_tuple(
              // 索引1（哈希索引）的构造参数
              boost::make_tuple(
                  0,  // bucket_count = 0，让 Boost 自动选择桶数

                  // 键提取器：从元素中提取 Pair（即 pair.first）
                  boost::multi_index::member<std::pair<Pair, size_t>,
                                             Pair,
                                             &std::pair<Pair, size_t>::first>(),
                  boost::hash<Pair>(),       // 哈希函数

                  std::equal_to<Pair>()),    // 键相等判断

              // 索引2（有序索引）的构造参数
              boost::make_tuple(
                  // 键提取器：元素整体作为键
                  boost::multi_index::identity<std::pair<Pair, size_t>>(),
                  // 比较器：我们自定义的排序规则（按计数降序）
                  PairCountCompare(vocab)))) {
        // 初始化词表的前 256 个 token：单字节字符
        for (size_t i = 0; i < 256; i++) {
            AddToken(std::string(1, static_cast<char>(i)));
        }
        // reserve 预留容量，避免后续 push_back 时频繁重新分配内存
        // 提前分配足够空间可以提升性能（减少内存再分配和元素拷贝）
        vocab.reserve(target_vocab_size);
    };

    /**
     * 析构函数（Destructor）
     *
     * 当对象生命周期结束时自动调用。
     * = default 表示使用编译器默认生成的析构函数（不做特殊清理）。
     * 因为所有成员变量（vector, unordered_map 等）都有各自的析构函数，
     * 会自动释放它们的资源（RAII 原则：资源获取即初始化）。
     */
    ~TokenizerTrainerC() = default;

    /**
     * LoadData：加载预分词数据
     *
     * 参数类型解析：
     *   const std::vector<py::bytes>& special_tokens：
     *     py::bytes 是 pybind11 的类型，表示 Python 的 bytes 对象。
     *     const & 表示常量引用，接收参数时不拷贝（只传引用）。
     *
     *   py::dict pre_token_count：
     *     pybind11 的 Python dict 映射类型。
     *     注意这里没有 &，是按值传递 py::dict。
     *     pybind11 的类型实际上是轻量级的引用计数句柄，拷贝代价很低（类似 Python 中的赋值）。
     *
     * 逻辑说明：
     *   遍历每个 pre-token，把它拆成字节序列，统计每对相邻字节的出现次数。
     */
    void LoadData(const std::vector<py::bytes>& special_tokens,
                  py::dict pre_token_count) {
        // 先添加特殊 token（如 <|endoftext|>）到词表
        // auto：C++11 的类型推导关键字，编译器根据右侧表达式自动推断类型。
        // 这里 const auto& 等价于 const py::bytes&（对 pybind11 类型按引用遍历）
        for (const auto& sp_token : special_tokens) {
            // .cast<std::string>() 将 py::bytes 转换为 C++ 的 std::string
            AddToken(sp_token.cast<std::string>());
        }

        // 预留空间：避免后续 emplace_back 时频繁重分配
        pre_token_state.reserve(pre_token_count.size());

        // 遍历 Python 字典 pre_token_count
        // for (const auto& item : pre_token_count)：范围 for 循环（range-based for），
        // C++11 引入，类似 Python 的 for item in dict.items()。
        // 对 py::dict 迭代，item 是 key-value 对。
        for (const auto& item : pre_token_count) {
            // py::isinstance<py::bytes>(item.first)：检查 key 是否为 Python bytes 类型
            if (!py::isinstance<py::bytes>(item.first)) {
                throw std::runtime_error(
                    "Pre-token must be of type bytes/string.");
            }

            // std::string_view：C++17 引入的"字符串视图"类型。
            // 与 std::string 不同，string_view 不拥有内存，只是指向已有字符串的"窗口"，
            // 因此构造和拷贝非常轻量（只存了指针+长度）。
            // 这里转为 string_view 后遍历其字符，避免构造 std::string 的开销。
            std::string_view pre_token =
                item.first.cast<py::bytes>().cast<std::string_view>();
            size_t count = item.second.cast<size_t>();  // 出现次数

            // 将 pre-token 的每个字节转为对应的 token 索引
            // 前 256 个 token 就是字节值本身（i.e., 索引 0='\x00', 65='A', 104='h'）
            std::vector<Index> pre_token_index;
            pre_token_index.reserve(pre_token.size());

            // unsigned char：确保是 0-255 无符号值
            // 这里用 unsigned char 而不是普通 char，避免 char 被解释为有符号数
            for (unsigned char c : pre_token) {
                pre_token_index.push_back(static_cast<Index>(c));
            }

            // 统计 pre-token 内部所有相邻 pair 的频次
            // size_t i = 0; i + 1 < size：确保至少有 2 个元素才会进入（i 和 i+1 都有效）
            for (size_t i = 0; i + 1 < pre_token_index.size(); i++) {
                Pair pair =
                    MakePair(pre_token_index[i], pre_token_index[i + 1]);
                // 更新 pair 的计数，加 count（因为这个 pre-token 出现了 count 次）
                UpdatePairCount(pair, count);

                // 记录"哪个 pre-token 包含了这个 pair"，后续合并时需要知道
                // static_cast<Index>(pre_token_state.size())：
                //   当前 pre_token_state 的大小就是即将添加的这个 pre-token 的索引
                //   （因为下面马上要 emplace_back 了，添加后索引就是当前 size）
                pair_to_containing_pre_tokens_index[pair].push_back(
                    static_cast<Index>(pre_token_state.size()));
            }

            // emplace_back：在 vector 末尾原地构造一个元素（避免临时对象的拷贝）
            // std::move 是移动语义：把 pre_token_index 的内容"移动"到 PreTokenState 中，
            // 而不是拷贝一份，避免 vector 的内存分配和数据复制。
            // std::move 之后，pre_token_index 变为"空"状态（被移动后的对象仍是有效的，但内容未定义）。
            pre_token_state.emplace_back(
                PreTokenState{std::move(pre_token_index), count});
        }
    }

    /**
     * DetermineMergePair：确定本轮要合并的 token pair
     *
     * 从有序索引（按计数降序排列）中取出计数最高的 pair。
     *
     *   pair_to_count.empty()：检查容器是否为空
     *
     *   pair_to_count.get<1>()：
     *     获取容器的第 2 个索引（索引从 0 开始计数）。
     *     get<by_count>() 也可以按标签名获取（但这里用序号 <1>）。
     *
     *   begin()->first：
     *     begin() 返回指向有序索引第一个元素的迭代器（类似指针）。
     *     第一个元素就是计数最高的 pair（因为排序规则是降序）。
     *     -> 是成员访问运算符，用于通过迭代器/指针访问成员（等价于 (*it).first）。
     *
     *   erase(...)：删除这个 pair（下次就不会再选它了）
     *
     * 返回值 0：当容器为空时返回 0（无效的 pair 值），表示没有更多可合并的 pair
     */
    Pair DetermineMergePair() {
        if (pair_to_count.empty()) return 0;
        Pair max_pair = pair_to_count.get<1>().begin()->first;
        pair_to_count.get<1>().erase(pair_to_count.get<1>().begin());
        return max_pair;
    }

    /**
     * UpdatePairCount：更新某个 pair 的计数
     *
     *   ssize_t delta：有符号的 size_t（可能为负数，表示减少计数）
     *
     *   auto it = pair_to_count.find(pair)：
     *     在哈希索引中查找 pair（O(1) 平均）。auto 自动推导为迭代器类型。
     *
     *   pair_to_count.modify(it, [&](auto& entry) { entry.second += delta; })：
     *     modify 是 Boost 多索引容器提供的函数，用于修改已有元素。
     *     直接改 second（计数），容器会自动维护其他索引（如有序索引会重新排序）。
     *
     *     参数中的 [&](auto& entry) { ... } 是 lambda 表达式：
     *       [&] 表示以引用方式捕获外部变量（可以访问函数内的所有变量）
     *       (auto& entry) 是参数列表（auto 表示类型由编译器推导）
     *       { entry.second += delta; } 是函数体
     *
     *   如果没找到 key，则用 insert 插入新元素：
     *     std::make_pair(pair, delta) 构造一个 pair<Pair, size_t>
     */
    void UpdatePairCount(Pair pair, ssize_t delta) {
        auto it = pair_to_count.find(pair);
        if (it != pair_to_count.end()) {  // 找到了，修改已有元素
            pair_to_count.modify(it,
                                 [&](auto& entry) { entry.second += delta; });
        } else {  // 没找到，插入新元素
            pair_to_count.insert(std::make_pair(pair, delta));
        }
    }

    /**
     * MergePair：执行一次 BPE 合并操作
     *
     * 这是 BPE 算法的核心步骤：
     *   1. 将两个 token 合并为一个新的 token（如 "l" + "l" → "ll"）
     *   2. 遍历所有包含该 pair 的 pre-token，把这两个相邻 token 替换为新 token
     *   3. 同时更新受影响的新的相邻 pair 的计数
     *
     * 详细注释见函数体内。
     */
    void MergePair(const Pair pair) {
        // 将两个 token 字符串拼接得到新 token 的字符串表示
        // vocab[GetFirst(pair)]：根据索引从词表中取出 first token 的字符串
        std::string merged_token =
            vocab[GetFirst(pair)] + vocab[GetSecond(pair)];
        AddToken(merged_token);

        // 新 token 的索引 = 词表大小 - 1（因为是最后添加的）
        Index new_index = static_cast<Index>(vocab.size() - 1);

        // 检查这个 pair 是否存在于映射表中（在某些边界情况可能不存在）
        // pair_to_containing_pre_tokens_index.find(key)：
        //   在 unordered_map 中查找 key，返回迭代器。
        //   如果没找到，返回 .end()（末尾哨兵迭代器）。
        if (pair_to_containing_pre_tokens_index.find(pair) ==
            pair_to_containing_pre_tokens_index.end()) {
            return;
        }

        // 获取所有包含此 pair 的 pre-token 的索引列表
        // 注意这里用的是引用常量 &，不拷贝整个 vector
        const std::vector<Index>& affected_pre_token_indexes =
            pair_to_containing_pre_tokens_index[pair];

        // 遍历每个受影响的 pre-token，在其中执行合并替换
        for (const Index pre_token : affected_pre_token_indexes) {
            PreTokenState& state = pre_token_state[pre_token];  // 获取该 pre-token 的状态
            size_t read_i = 0;   // 读取指针：从 pre-token 序列读取原始 token
            size_t write_i = 0;  // 写入指针：写入合并后的 token 序列（原地操作，写入位置可能滞后于读取）
            size_t pair_count = 0;  // 该 pre-token 中被合并的 pair 个数

            // 遍历 pre-token 序列中的所有 token
            for (; read_i < state.pre_token_index.size(); read_i++) {
                // 处理最后一个 token：不可能和下一个组成 pair
                if (read_i == state.pre_token_index.size() - 1) {
                    state.pre_token_index[write_i++] =
                        state.pre_token_index[read_i];  // 直接拷贝
                    continue;  // 跳过本次循环剩余部分，进入下一次迭代
                }

                // 构造当前 token 和下一个 token 组成的 pair
                Pair current_pair = MakePair(state.pre_token_index[read_i],
                                             state.pre_token_index[read_i + 1]);
                if (current_pair != pair) {
                    // 不是要合并的 pair，直接拷贝当前 token 到写入位置
                    state.pre_token_index[write_i++] =
                        state.pre_token_index[read_i];
                    continue;
                }

                // === 找到了要合并的 pair！执行合并操作 ===
                pair_count++;

                // 处理合并后的左邻接 pair（如果存在）
                // 例如序列 ... A B C ... 合并 B+C → D 后，需要：
                //   删除旧 pair (A, B)，添加新 pair (A, D)
                if (write_i > 0) {  // 如果前面还有 token
                    // 旧的左邻 pair (前一个token, 当前被合并的第一个token)
                    Pair old_left_pair =
                        MakePair(state.pre_token_index[write_i - 1],
                                 state.pre_token_index[read_i]);
                    UpdatePairCount(old_left_pair, -state.count);  // 减去权重

                    // 新的左邻 pair (前一个token, 新合并的token)
                    Pair new_left_pair =
                        MakePair(state.pre_token_index[write_i - 1], new_index);
                    UpdatePairCount(new_left_pair, state.count);  // 加上权重

                    // 记录新 pair 与 pre-token 的映射关系
                    pair_to_containing_pre_tokens_index[new_left_pair]
                        .push_back(pre_token);
                }

                // 处理合并后的右邻接 pair（如果存在）
                // 例如序列 ... B C D ... 合并 B+C → E 后，需要：
                //   删除旧 pair (C, D)，添加新 pair (E, D)
                if (read_i + 2 < state.pre_token_index.size()) {  // 如果后面还有 token
                    // 旧的右邻 pair (被合并的第二个token, 下一个token)
                    Pair old_right_pair =
                        MakePair(state.pre_token_index[read_i + 1],
                                 state.pre_token_index[read_i + 2]);
                    UpdatePairCount(old_right_pair, -state.count);  // 减去权重

                    // 新的右邻 pair (新合并的token, 下一个token)
                    Pair new_right_pair =
                        MakePair(new_index, state.pre_token_index[read_i + 2]);
                    UpdatePairCount(new_right_pair, state.count);  // 加上权重

                    pair_to_containing_pre_tokens_index[new_right_pair]
                        .push_back(pre_token);
                }

                // 将新 token 的索引写入
                state.pre_token_index[write_i++] = new_index;
                read_i++;  // 跳过被合并的第二个 token（因为它已经被合并到 new_index 中了）
            }

            // 如果此 pre-token 中有合并发生，需要调整序列长度
            if (pair_count > 0) {
                // 验证：原始长度 - 合并对数 == 新长度
                // 因为每合并一对，两个 token 变成一个，长度减少 1
                // assert 是调试用的断言宏，如果条件为 false 则程序报错终止
                // 仅在 Debug 模式下生效，Release 模式下被编译器忽略
                assert(write_i + pair_count == state.pre_token_index.size());
                // resize：调整 vector 大小，截断多余的元素（合并后序列变短了）
                state.pre_token_index.resize(state.pre_token_index.size() -
                                             pair_count);
            }
        }

        // 清除已被合并的 pair 的各种记录，因为该 pair 不会再出现在任何 pre-token 中了
        pair_to_count.erase(pair);                            // 从计数容器中删除
        pair_to_containing_pre_tokens_index.erase(pair);      // 从映射表中删除
    }

    /**
     * train：执行 BPE 训练的主循环
     *
     * 返回值类型：
     *   std::tuple<std::unordered_map<Index, py::bytes>,
     *              std::vector<std::tuple<py::bytes, py::bytes>>>
     *
     *   这是一个 tuple（元组），包含两个元素：
     *     元素1：从 token 索引到 bytes 的映射（词表）
     *     元素2：merge 操作的序列，每个元素是 (token_a, token_b) 表示 token_a + token_b → 新token
     *
     * 返回类型解构：
     *   std::tuple<A, B> = 打包两个不同类型的值
     *   std::unordered_map<K, V> = 哈希映射（无序字典），K 为键类型，V 为值类型
     *   std::vector<T> = 动态数组
     *   py::bytes = Python bytes 的 C++ 对应类型
     */
    std::tuple<std::unordered_map<Index, py::bytes>,
               std::vector<std::tuple<py::bytes, py::bytes>>>
    train() {
        // 计算还需要执行多少次合并
        // target_vocab_size - vocab.size()：目标词表大小 - 当前词表大小
        size_t loops = target_vocab_size - vocab.size();

        // 主循环：每次迭代选择并合并最高频的 pair
        // && 是逻辑与运算符（短路求值：左边为 false 就不执行右边）
        // !pair_to_count.empty()：还有 pair 可以合并
        while (loops > 0 && !pair_to_count.empty()) {
            Pair merge_pair = DetermineMergePair();  // 选出最高频 pair
            merges.push_back(merge_pair);            // 记录 merge 顺序（push_back：向 vector 末尾添加元素）
            MergePair(merge_pair);                   // 执行合并操作
            loops--;                                  // 剩余循环次数减 1
        }

        // 构建索引→token 的映射（词表）
        // 不用在构造函数里 auto，这里显式写了类型以便理解
        std::unordered_map<Index, py::bytes> index_to_token;
        for (Index i = 0; i < vocab.size(); i++) {
            // py::bytes(vocab[i])：将 C++ string 转换为 Python bytes 对象
            index_to_token[i] = vocab[i];
        }

        // 构建 merge 序列（token_a, token_b）列表
        // 记录了每个合并操作的先后顺序（训练好的 tokenizer 解码时需要按此顺序执行）
        std::vector<std::tuple<py::bytes, py::bytes>> token_merges;
        token_merges.reserve(merges.size());

        for (const Pair& p : merges) {
            // emplace_back：在末尾原地构造元素（直接传入构造参数，不需要创建临时对象）
            token_merges.emplace_back(py::bytes(vocab[GetFirst(p)]),
                                      py::bytes(vocab[GetSecond(p)]));
        }

        // std::make_tuple：创建 tuple，自动推导类型
        // 等价于 return {index_to_token, token_merges};
        return std::make_tuple(index_to_token, token_merges);
    }

   private:
    /**
     * AddToken：向词表添加一个新 token
     *
     * private 成员函数：外部代码无法调用，仅类内部使用。
     *
     * std::move(token)：
     *   移动语义。参数 token 是传值进来的（会拷贝一次），
     *   然后用 std::move 把拷贝的内容"移动"到 vocab 中，避免再次拷贝。
     *   对于 std::string，移动操作只是转移内部指针，非常快（O(1)）。
     */
    void AddToken(std::string token) { vocab.push_back(std::move(token)); }

    // -------------------------------------------------------------------------
    // 成员变量（Member Variables）
    // -------------------------------------------------------------------------
    // private 成员变量：封装原则，外部不能直接访问，只能通过公有接口操作。

   private:
    std::vector<std::string> vocab;           // 词表：按索引存储 token 的字符串
    std::vector<PreTokenState> pre_token_state;  // 所有 pre-token 的状态（token序列 + 频次）
    PairCountContainer pair_to_count;         // (token对 → 计数) 多索引容器（同时支持哈希查找和排序）
    std::unordered_map<Pair, std::vector<Index>>
        pair_to_containing_pre_tokens_index;  // 每个 pair 出现在哪些 pre-token 中
    std::vector<Pair> merges;                 // 按合并顺序记录的所有 merge pair
    size_t target_vocab_size;                 // 目标词表大小
};

// ===========================================================================
// pybind11 模块绑定（PYBIND11_MODULE）
// ===========================================================================
// PYBIND11_MODULE 是一个宏（macro），用于定义一个 Python 扩展模块。
// 宏参数：
//   - tokenizer_cpp：生成的 Python 模块名，即 import tokenizer_cpp
//   - m：模块对象变量名（在宏体内使用 m 来访问模块）
//
// 宏展开后大致相当于定义了一个名为 PyInit_tokenizer_cpp 的函数，
// Python 在 import 时会自动调用它来初始化模块。
//
PYBIND11_MODULE(tokenizer_cpp, m) {
    // m.doc()：设置模块的文档字符串（Python 中 __doc__ 属性）
    m.doc() = "BPE Tokenizer Trainer implemented in C++";

    // py::class_<CppClassName>(module, "PythonClassName")：
    //   将 C++ 类 TokenizerTrainerC 暴露为 Python 类，Python 中叫 "TokenizerTrainerC"
    //
    // .def：定义类的成员函数绑定
    //   链式调用（每次 .def 返回自己，可以继续调用）：
    //
    //   py::init<参数类型>()：绑定构造函数
    //     py::arg("参数名")：给 Python 端参数命名（便于 keyword argument 调用）
    //
    //   .def("方法名", &C++方法地址, py::arg("参数名")..., "文档字符串")
    //     &TokenizerTrainerC::LoadData：成员函数指针
    py::class_<TokenizerTrainerC>(m, "TokenizerTrainerC")
        .def(py::init<size_t>(), py::arg("target_vocab_size"))
        .def("LoadData",
             &TokenizerTrainerC::LoadData,
             py::arg("special_tokens"),
             py::arg("pre_token_count"),
             "Load pre-tokenized data and special tokens.")
        .def("train",
             &TokenizerTrainerC::train,
             "Train the tokenizer and return vocabulary and merges.");
}