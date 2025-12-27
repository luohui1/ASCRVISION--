"""
CableVision AI - 智能报告生成服务
集成大模型API自动生成专业质检报告
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional


# 缺陷信息映射
DEFECT_INFO = {
    0: {'name': '表面划伤', 'name_en': 'scratch', 'severity': 'high', 
        'cause': '生产过程中机械摩擦或运输碰撞', 'impact': '可能导致绝缘层破损，影响电缆使用寿命'},
    1: {'name': '绝缘气泡', 'name_en': 'bubble', 'severity': 'medium',
        'cause': '挤出过程中温度控制不当或原料含水', 'impact': '降低绝缘性能，可能引发局部放电'},
    2: {'name': '护套裂纹', 'name_en': 'crack', 'severity': 'high',
        'cause': '材料老化、温度应力或机械损伤', 'impact': '严重影响防护性能，需立即处理'},
    3: {'name': '凹陷变形', 'name_en': 'dent', 'severity': 'medium',
        'cause': '外力挤压或存储不当', 'impact': '可能影响电缆弯曲性能和信号传输'},
    4: {'name': '颜色异常', 'name_en': 'discolor', 'severity': 'low',
        'cause': '原料批次差异或加工温度波动', 'impact': '主要影响外观，一般不影响性能'},
    5: {'name': '印字缺失', 'name_en': 'print_miss', 'severity': 'medium',
        'cause': '印字设备故障或油墨附着不良', 'impact': '影响产品标识和追溯'},
    6: {'name': '偏心', 'name_en': 'eccentric', 'severity': 'high',
        'cause': '挤出模具偏移或牵引速度不稳', 'impact': '严重影响绝缘均匀性，可能导致击穿'},
    7: {'name': '杂质', 'name_en': 'impurity', 'severity': 'medium',
        'cause': '原料污染或生产环境不洁', 'impact': '可能形成电气薄弱点'},
    8: {'name': '褶皱', 'name_en': 'wrinkle', 'severity': 'low',
        'cause': '冷却不均匀或牵引张力不稳', 'impact': '影响外观，严重时影响弯曲性能'},
    9: {'name': '剥离', 'name_en': 'peel', 'severity': 'high',
        'cause': '层间粘接不良或材料相容性差', 'impact': '严重影响电缆结构完整性'},
}

# 质量等级定义
QUALITY_GRADES = {
    'A': {'name': '优等品', 'desc': '无任何缺陷，完全符合标准'},
    'B': {'name': '一等品', 'desc': '仅有轻微缺陷，不影响使用'},
    'C': {'name': '合格品', 'desc': '存在中等缺陷，需关注'},
    'D': {'name': '不合格', 'desc': '存在严重缺陷，需返工或报废'},
}


class AIReportGenerator:
    """AI智能报告生成器"""
    
    def __init__(self, api_key: str = None, api_type: str = 'qwen'):
        """
        初始化报告生成器
        api_type: 'qwen' (通义千问), 'openai', 'wenxin' (文心一言)
        """
        self.api_key = api_key or os.getenv('AI_API_KEY')
        self.api_type = api_type
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化API客户端"""
        if not self.api_key:
            print("警告: 未配置API密钥，将使用本地模板生成报告")
            return
        
        try:
            if self.api_type == 'qwen':
                import dashscope
                dashscope.api_key = self.api_key
                self.client = dashscope
            elif self.api_type == 'openai':
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            print(f"✓ {self.api_type} API客户端初始化成功")
        except ImportError as e:
            print(f"警告: 缺少依赖包 - {e}")
            self.client = None
    
    def analyze_defects(self, detection_result: Dict) -> Dict:
        """分析检测结果，生成缺陷分析报告"""
        defects = detection_result.get('defects', [])
        
        # 统计缺陷
        severity_count = {'high': 0, 'medium': 0, 'low': 0}
        defect_types = {}
        
        for d in defects:
            sev = d.get('severity', 'unknown')
            if sev in severity_count:
                severity_count[sev] += 1
            
            cls_name = d.get('class_name', '未知')
            defect_types[cls_name] = defect_types.get(cls_name, 0) + 1
        
        # 确定质量等级
        if len(defects) == 0:
            grade = 'A'
        elif severity_count['high'] > 0:
            grade = 'D'
        elif severity_count['medium'] > 2:
            grade = 'C'
        elif severity_count['medium'] > 0 or severity_count['low'] > 3:
            grade = 'B'
        else:
            grade = 'A'
        
        return {
            'total_defects': len(defects),
            'severity_count': severity_count,
            'defect_types': defect_types,
            'quality_grade': grade,
            'grade_info': QUALITY_GRADES[grade],
            'is_qualified': grade != 'D'
        }
    
    def generate_report(self, detection_result: Dict, 
                       sample_info: Dict = None) -> Dict:
        """生成完整的质检报告"""
        analysis = self.analyze_defects(detection_result)
        
        # 基础报告信息
        report = {
            'report_id': f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'sample_info': sample_info or {},
            'detection_summary': {
                'total_defects': analysis['total_defects'],
                'defect_types': analysis['defect_types'],
                'severity_distribution': analysis['severity_count'],
            },
            'quality_assessment': {
                'grade': analysis['quality_grade'],
                'grade_name': analysis['grade_info']['name'],
                'grade_description': analysis['grade_info']['desc'],
                'is_qualified': analysis['is_qualified'],
            },
            'defect_details': [],
            'ai_analysis': '',
            'recommendations': [],
        }
        
        # 添加缺陷详情
        for d in detection_result.get('defects', []):
            cls_id = d.get('class_id', 0)
            info = DEFECT_INFO.get(cls_id, {})
            report['defect_details'].append({
                'type': d.get('class_name', '未知'),
                'confidence': d.get('confidence', 0),
                'severity': d.get('severity', 'unknown'),
                'bbox': d.get('bbox', []),
                'cause': info.get('cause', '未知'),
                'impact': info.get('impact', '未知'),
            })
        
        # 生成AI分析
        if self.client:
            report['ai_analysis'] = self._call_ai_analysis(report)
        else:
            report['ai_analysis'] = self._generate_local_analysis(report)
        
        # 生成建议
        report['recommendations'] = self._generate_recommendations(analysis)
        
        return report
    
    def _call_ai_analysis(self, report: Dict) -> str:
        """调用大模型API生成分析"""
        prompt = self._build_analysis_prompt(report)
        
        try:
            if self.api_type == 'qwen':
                from dashscope import Generation
                response = Generation.call(
                    model='qwen-turbo',
                    prompt=prompt,
                    max_tokens=800,
                )
                return response.output.text if response.output else self._generate_local_analysis(report)
            
            elif self.api_type == 'openai':
                response = self.client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=800,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"AI分析调用失败: {e}")
            return self._generate_local_analysis(report)
    
    def _build_analysis_prompt(self, report: Dict) -> str:
        """构建AI分析提示词"""
        defects_desc = []
        for d in report['defect_details']:
            defects_desc.append(f"- {d['type']}: 置信度{d['confidence']:.1%}, 严重程度{d['severity']}")
        
        defects_text = '\n'.join(defects_desc) if defects_desc else '无缺陷'
        
        return f"""你是一位专业的电缆质检工程师。请根据以下AI检测结果，生成专业的质检分析报告。

检测结果:
- 缺陷总数: {report['detection_summary']['total_defects']}
- 质量等级: {report['quality_assessment']['grade_name']}
- 缺陷详情:
{defects_text}

请从以下几个方面进行分析（200字以内）:
1. 整体质量评估
2. 主要问题分析
3. 可能的产生原因
4. 对产品性能的影响

请用专业、简洁的语言回答。"""
    
    def _generate_local_analysis(self, report: Dict) -> str:
        """本地模板生成分析（无API时使用）"""
        grade = report['quality_assessment']['grade']
        total = report['detection_summary']['total_defects']
        
        if grade == 'A':
            return "本批次电缆样品经AI视觉检测系统全面检查，未发现任何表面缺陷，各项指标均符合质量标准要求。产品外观完整，绝缘层均匀，护套无损伤，建议正常出厂。"
        
        elif grade == 'B':
            return f"本批次电缆样品检测发现{total}处轻微缺陷，主要为外观类问题，不影响产品电气性能和使用安全。建议在包装前进行外观复检，确认缺陷位置并做好标记，可正常出厂使用。"
        
        elif grade == 'C':
            defect_types = list(report['detection_summary']['defect_types'].keys())
            return f"本批次电缆样品检测发现{total}处缺陷，包括{', '.join(defect_types)}等问题。部分缺陷可能影响产品长期使用性能，建议进行人工复检确认，必要时进行局部修复处理后方可出厂。"
        
        else:  # grade == 'D'
            defect_types = list(report['detection_summary']['defect_types'].keys())
            return f"本批次电缆样品检测发现{total}处严重缺陷，包括{', '.join(defect_types)}等问题。存在影响产品安全性和可靠性的重大隐患，建议立即隔离该批次产品，进行全面质量追溯，查明原因后进行返工或报废处理。"
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成处理建议"""
        recommendations = []
        grade = analysis['quality_grade']
        severity = analysis['severity_count']
        defect_types = analysis['defect_types']
        
        if grade == 'A':
            recommendations.append("产品质量优良，建议正常入库出厂")
            recommendations.append("继续保持当前生产工艺参数")
        
        elif grade == 'B':
            recommendations.append("建议进行人工外观复检确认")
            recommendations.append("记录缺陷位置，便于后续追溯")
            if '颜色异常' in defect_types:
                recommendations.append("检查原料批次一致性")
        
        elif grade == 'C':
            recommendations.append("暂停该批次出厂，进行全面复检")
            recommendations.append("分析缺陷产生原因，调整工艺参数")
            if severity['medium'] > 1:
                recommendations.append("建议增加在线检测频率")
        
        else:  # grade == 'D'
            recommendations.append("立即隔离该批次产品")
            recommendations.append("启动质量追溯程序，查明根本原因")
            recommendations.append("检查生产设备状态，必要时停机维护")
            if '偏心' in defect_types or '剥离' in defect_types:
                recommendations.append("重点检查挤出机模具和牵引系统")
            if '裂纹' in defect_types or '划伤' in defect_types:
                recommendations.append("检查冷却系统和传输辊道")
        
        return recommendations
    
    def export_to_json(self, report: Dict, filepath: str) -> str:
        """导出报告为JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return filepath
    
    def export_to_markdown(self, report: Dict, filepath: str = None) -> str:
        """导出报告为Markdown格式"""
        md = f"""# 电缆质检报告

## 基本信息
- **报告编号**: {report['report_id']}
- **检测时间**: {report['timestamp']}
- **质量等级**: {report['quality_assessment']['grade']} - {report['quality_assessment']['grade_name']}

## 检测结果摘要
- **缺陷总数**: {report['detection_summary']['total_defects']}
- **严重缺陷**: {report['detection_summary']['severity_distribution']['high']}
- **中等缺陷**: {report['detection_summary']['severity_distribution']['medium']}
- **轻微缺陷**: {report['detection_summary']['severity_distribution']['low']}

## AI分析
{report['ai_analysis']}

## 缺陷详情
"""
        for i, d in enumerate(report['defect_details'], 1):
            md += f"""
### 缺陷 {i}: {d['type']}
- 置信度: {d['confidence']:.1%}
- 严重程度: {d['severity']}
- 可能原因: {d['cause']}
- 影响: {d['impact']}
"""
        
        md += "\n## 处理建议\n"
        for i, rec in enumerate(report['recommendations'], 1):
            md += f"{i}. {rec}\n"
        
        md += f"\n---\n*本报告由 CableVision AI 智能质检系统自动生成*"
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md)
        
        return md
