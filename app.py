import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="历史人格画像测试",
    page_icon="🏛️",
    layout="wide"
)

# ==================== 加载画像中心点 ====================
@st.cache_data
def load_centers():
    df = pd.read_excel("new/cluster_centers_8d.xlsx", index_col=0)
    return df

centers_df = load_centers()
画像列表 = centers_df.index.tolist()

# ==================== 加载情景题 ====================
@st.cache_data
def load_scenarios():
    with open("new/generated_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return scenarios

scenarios = load_scenarios()

# 维度列表
dimensions = ["战略扩张", "民生治国", "文化文治", "集权秩序", 
              "外交合作", "纳谏用人", "勤政实干", "传承规划"]

# max_scores（根据你的实际题目计算）
max_scores = {
    "战略扩张": 138,
    "民生治国": 166,
    "文化文治": 143,
    "集权秩序": 187,
    "外交合作": 123,
    "纳谏用人": 157,
    "勤政实干": 162,
    "传承规划": 138,
}

# ==================== 初始化 Session State ====================
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "scores" not in st.session_state:
    st.session_state.scores = {dim: 0 for dim in dimensions}
if "finished" not in st.session_state:
    st.session_state.finished = False
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== 计分函数 ====================
def add_scores(scores_dict):
    for dim, val in scores_dict.items():
        if dim in st.session_state.scores:
            st.session_state.scores[dim] += val

def next_question():
    st.session_state.current_q += 1
    if st.session_state.current_q >= len(scenarios):
        st.session_state.finished = True

def reset_test():
    st.session_state.current_q = 0
    st.session_state.scores = {dim: 0 for dim in dimensions}
    st.session_state.finished = False
    st.session_state.history = []

# ==================== 计算用户向量 ====================
def compute_user_vector():
    """将用户得分标准化到 0-10 范围"""
    user_vec = []
    for dim in dimensions:
        raw_score = st.session_state.scores.get(dim, 0)
        max_score = max_scores.get(dim, 200)
        if max_score > 0:
            normalized = (raw_score / max_score) * 10
        else:
            normalized = 5
        normalized = max(0.0, min(10.0, normalized))
        user_vec.append(normalized)
    return np.array(user_vec).reshape(1, -1)

# ==================== 匹配函数 ====================
def match_profile():
    if centers_df.empty:
        return "无法匹配", 0, []
    
    user_vec = compute_user_vector().flatten()[:8]
    
    best_profile = None
    best_corr = -1
    results = []
    
    # 正确遍历 DataFrame
    for profile_name, row in centers_df.iterrows():
        center_vec = row.iloc[:8].values
        corr, _ = spearmanr(user_vec, center_vec)
        if np.isnan(corr):
            corr = 0
        results.append((profile_name, corr))
        if corr > best_corr:
            best_corr = corr
            best_profile = profile_name
    
    results.sort(key=lambda x: x[1], reverse=True)
    best_score = (best_corr + 1) / 2
    
    return best_profile, best_score, results

# ==================== 画像描述 ====================
profile_descriptions = {
    "德善守成型": {
        "desc": "重视民生、善于用人、勤政实干。是团队中值得信赖的建设者和管理者。",
        "strengths": "用人有道，关注实务",
        "weaknesses": "开拓精神稍弱",
        "examples": "汉文帝、刘备"
    },
    "昏庸无能型": {
        "desc": "倾向于回避责任，在重大决策上缺乏判断力，已被权臣左右，丧权辱国。",
        "strengths": "心态放松，不易焦虑",
        "weaknesses": "决策能力不足，易被误导",
        "examples": "宋襄公、楚怀王"
    },
    "保守平庸型": {
        "desc": "能力一般，保守求稳，没有太多建树，但也没有大的过失。",
        "strengths": "稳重、不犯大错",
        "weaknesses": "缺乏进取心，碌碌无为",
        "examples": "魏惠王、万历"
    },
    "全能贤主型": {
        "desc": "全面发展，既有战略眼光又关心民生，善于用人且执行力强。是天生的卓越领导者。",
        "strengths": "全面均衡，无短板",
        "weaknesses": "标准太高，可能对他人要求苛刻",
        "examples": "唐太宗、康熙"
    },
    "暴虐亡国型": {
        "desc": "性格暴虐，不恤民力，最终导致国家衰败或灭亡。",
        "strengths": "有魄力，敢作敢为",
        "weaknesses": "残暴、不听劝、不顾民生",
        "examples": "项羽、尼禄"
    },
    "军事统帅型": {
        "desc": "崇尚实力、战略清晰、执行力强。是天生的竞争者和军事领袖。",
        "strengths": "战略思维，果断行动",
        "weaknesses": "可能忽视文化建设和长期软实力",
        "examples": "齐桓公、成吉思汗"
    },
    "文武兼修型": {
        "desc": "文武双全，既能制定战略又有文化修养，重视国际形象。是天生的外交家和战略家。",
        "strengths": "文化素养高，国际视野开阔",
        "weaknesses": "对民生细节关注较少",
        "examples": "汉武帝、曹操"
    },
    "文治误国型": {
        "desc": "文化艺术修养高，但治国能力差，沉迷个人爱好误国。",
        "strengths": "文化素养高，有艺术天赋",
        "weaknesses": "不懂治国，不理朝政",
        "examples": "李煜、宋徽宗"
    },
    "改革集权型": {
        "desc": "强势改革，善于集权，有魄力和执行力，但可能手段强硬。",
        "strengths": "改革魄力，集权效率",
        "weaknesses": "手段激进，可能得罪既得利益",
        "examples": "秦始皇、隋文帝"
    }
}

# ==================== UI ====================
st.title("🏛️ 历史人格画像测试")
st.markdown("假设你是身处历史关键时刻的君主，你会如何抉择？")

if not st.session_state.finished:
    progress = st.session_state.current_q / len(scenarios)
    st.progress(progress, text=f"进度：{st.session_state.current_q}/{len(scenarios)}")

if st.session_state.finished:
    if not centers_df.empty:
        best_profile, match_score, all_results = match_profile()
        
        st.balloons()
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"🎯 您最匹配：{best_profile}")
            st.progress(match_score, text=f"匹配度：{match_score:.1%}")
            
            desc = profile_descriptions.get(best_profile, {})
            st.markdown(f"**📖 核心特征**：{desc.get('desc', '')}")
            st.markdown(f"**💪 优势倾向**：{desc.get('strengths', '')}")
            st.markdown(f"**⚠️ 潜在提醒**：{desc.get('weaknesses', '')}")
            st.markdown(f"**👑 代表人物**：{desc.get('examples', '')}")
        
        with col2:
            st.markdown("**📊 你的维度得分**")
            user_vec = compute_user_vector().flatten()
            for i, dim in enumerate(dimensions):
                st.markdown(f"- {dim}: {user_vec[i]:.1f}/10.0")
        
        with st.expander("📊 查看所有画像匹配度"):
            for profile, score in all_results:
                st.write(f"- {profile}: {(score + 1) / 2:.1%}")
    else:
        st.error("无法加载画像数据，请检查 cluster_centers_8d.xlsx 文件")
    
    if st.button("🔄 重新测试"):
        reset_test()
        st.rerun()
    
    st.caption("本测试基于历史君主多维评分模型 | GMM聚类分析 | 斯皮尔曼匹配")

else:
    q = scenarios[st.session_state.current_q]
    
    with st.container():
        st.subheader(f"情景 {q['id']}：{q['title']}")
        st.markdown(f"> {q['background']}")
        
        st.markdown("---")
        
        cols = st.columns(2)
        for idx, opt in enumerate(q['options']):
            col = cols[idx % 2]
            with col:
                if st.button(f"{chr(65+idx)}. {opt['text']}", key=f"q{q['id']}_{idx}", use_container_width=True):
                    add_scores(opt['scores'])
                    st.session_state.history.append({
                        "情景": q['id'],
                        "选择": opt['text'],
                        "得分": opt['scores']
                    })
                    next_question()
                    st.rerun()
    
    st.divider()
    st.caption(f"第 {st.session_state.current_q + 1}/{len(scenarios)} 题 | 每个选择都会影响最终画像")

st.divider()
st.caption("© 历史人格画像测试 | 基于历史君主多维评分模型 | GMM聚类分析 | 斯皮尔曼匹配")
