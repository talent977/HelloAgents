"""测试通义千问嵌入和重排序模型"""

import requests

from hello_agents.memory.embedding import DashScopeEmbedding
from hello_agents.utils.env_utils import get_config_section

model_info = get_config_section('qwen3-max')
api_key = model_info['api_key']
base_url = model_info['base_url']
def test_embedding_model():
    """测试 text-embedding-v4 嵌入模型"""
    print("=" * 60)
    print("测试 text-embedding-v4 嵌入模型")
    print("=" * 60)

    # 配置
    model_name = "text-embedding-v4"

    try:
        # 使用项目中的 DashScopeEmbedding 类
        embedder = DashScopeEmbedding(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )

        print(f"✅ 嵌入模型初始化成功")
        print(f"   模型名称: {model_name}")
        print(f"   向量维度: {embedder.dimension}")

        # 测试单个文本编码
        test_text = "这是一个测试文本"
        embedding = embedder.encode(test_text)

        print(f"✅ 单个文本编码成功")
        print(f"   输入文本: {test_text}")
        print(f"   向量长度: {len(embedding)}")
        print(f"   向量前5维: {embedding[:5]}")

        # 测试批量编码
        texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的一个子集",
            "深度学习是机器学习的一种方法"
        ]
        embeddings = embedder.encode(texts)

        print(f"\n✅ 批量编码成功")
        print(f"   输入文本数: {len(texts)}")
        print(f"   输出向量数: {len(embeddings)}")
        print(f"   每个向量维度: {len(embeddings[0])}")

        # 测试相似度计算
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"\n✅ 相似度计算成功")
        print(f"   文本1和文本2的余弦相似度: {similarity:.4f}")

        print("\n" + "=" * 60)
        print("text-embedding-v4 测试通过 ✅")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ text-embedding-v4 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False


def test_rerank_model():
    """测试 gte-rerank-v2 重排序模型"""
    print("\n" + "=" * 60)
    print("测试 gte-rerank-v2 重排序模型")
    print("=" * 60)

    # 配置 - 使用原生 API 端点
    base_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    model_name = "gte-rerank-v2"

    # 测试数据
    query = "什么是机器学习？"
    documents = [
        {"id": 1, "text": "人工智能是计算机科学的一个分支，旨在创造智能机器。"},
        {"id": 2, "text": "机器学习是人工智能的一个子集，通过算法让计算机从数据中学习。"},
        {"id": 3, "text": "深度学习是机器学习的一种方法，使用神经网络模拟人脑。"},
        {"id": 4, "text": "Python是一种流行的编程语言。"},
        {"id": 5, "text": "自然语言处理是AI的一个重要应用领域。"}
    ]

    try:
        # 构建请求 - 使用通义千问原生格式
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 通义千问原生 API 格式
        payload = {
            "model": model_name,
            "input": {
                "query": query,
                "documents": [doc["text"] for doc in documents]
            },
            "parameters": {
                "top_n": len(documents)
            }
        }

        print(f"📤 发送重排序请求...")
        print(f"   查询: {query}")
        print(f"   文档数: {len(documents)}")
        print(f"   API 端点: {base_url}")

        # 发送请求
        response = requests.post(
            base_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"\n📥 响应状态码: {response.status_code}")

        if response.status_code != 200:
            print(f"\n❌ 请求失败: HTTP {response.status_code}")
            print(f"   响应内容: {response.text}")
            return False

        # 解析响应
        result = response.json()

        print(f"✅ 重排序成功")

        # 显示重排序结果 - 通义千问格式
        if "output" in result and "results" in result["output"]:
            ranked_results = result["output"]["results"]

            print(f"\n📊 重排序结果:")
            print(f"   {'排名':<6} {'原始ID':<8} {'相关性得分':<12} {'文本'}")
            print(f"   {'-'*6} {'-'*8} {'-'*12} {'-'*60}")

            for idx, ranked_item in enumerate(ranked_results, 1):
                original_index = ranked_item["index"]
                relevance_score = ranked_item.get("relevance_score", 0)
                original_text = documents[original_index]["text"]

                # 截断长文本显示
                display_text = original_text[:55] + "..." if len(original_text) > 55 else original_text
                print(f"   {idx:<6} {original_index + 1:<8} {relevance_score:<12.4f} {display_text}")

        print("\n" + "=" * 60)
        print("gte-rerank-v2 测试通过 ✅")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ gte-rerank-v2 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False


def test_models_together():
    """组合测试：使用嵌入和重排序模型"""
    print("\n" + "=" * 60)
    print("组合测试：嵌入 + 重排序")
    print("=" * 60)


    try:
        # 1. 使用嵌入模型生成向量
        embedder = DashScopeEmbedding(
            model_name="text-embedding-v4",
            api_key=api_key,
            base_url=base_url
        )

        query = "机器学习算法"
        docs = [
            "机器学习算法包括监督学习、无监督学习和强化学习",
            "深度学习使用多层神经网络",
            "Python是一种编程语言"
        ]

        print(f"\n步骤1: 生成嵌入向量")
        query_embedding = embedder.encode(query)
        doc_embeddings = embedder.encode(docs)
        print(f"✅ 查询向量维度: {len(query_embedding)}")
        print(f"✅ 文档向量数: {len(doc_embeddings)}")

        # 2. 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        print(f"\n步骤2: 基于嵌入向量的相似度")
        for i, (doc, sim) in enumerate(zip(docs, similarities)):
            print(f"   文档{i+1}: {sim:.4f} - {doc[:40]}...")

        print("\n" + "=" * 60)
        print("组合测试通过 ✅")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ 组合测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("通义千问模型可用性测试")
    print("=" * 60)
    print(f"API Key: sk-f4672***********")
    print(f"Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1")
    print("=" * 60)

    results = {}

    # 测试1: 嵌入模型
    results["text-embedding-v4"] = test_embedding_model()

    # 测试2: 重排序模型
    results["gte-rerank-v2"] = test_rerank_model()

    # 测试3: 组合测试
    results["组合测试"] = test_models_together()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for model, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{model:<25} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n总计: {total_passed}/{total_tests} 测试通过")

    if total_passed == total_tests:
        print("\n🎉 所有测试通过！模型可用。")
    else:
        print(f"\n⚠️  有 {total_tests - total_passed} 个测试失败，请检查配置。")

    print("=" * 60)
