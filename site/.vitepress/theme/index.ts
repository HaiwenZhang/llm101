import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import HomePage from './components/HomePage.vue'
import ProgressTracker from './components/ProgressTracker.vue'
import KVCacheLab from './components/KVCacheLab.vue'
import AttentionLab from './components/AttentionLab.vue'
import MoERouterLab from './components/MoERouterLab.vue'
import PaperLibrary from './components/PaperLibrary.vue'
import ConceptCheck from './components/ConceptCheck.vue'
import TokenLab from './components/TokenLab.vue'
import TrainingLoopLab from './components/TrainingLoopLab.vue'
import TransformerArchitecture from './components/TransformerArchitecture.vue'
import LearningStagesDiagram from './components/LearningStagesDiagram.vue'
import K3SystemDiagram from './components/K3SystemDiagram.vue'
import SamplingLab from './components/SamplingLab.vue'
import RAGPipelineLab from './components/RAGPipelineLab.vue'
import VectorSimilarityLab from './components/VectorSimilarityLab.vue'
import BPELab from './components/BPELab.vue'
import ScalingLab from './components/ScalingLab.vue'
import QuantizationLab from './components/QuantizationLab.vue'
import PreferenceRLLab from './components/PreferenceRLLab.vue'
import ChapterReadings from './components/ChapterReadings.vue'
import RecurrentGradientLab from './components/RecurrentGradientLab.vue'
import ModernDecoderLab from './components/ModernDecoderLab.vue'
import TestTimeScalingLab from './components/TestTimeScalingLab.vue'
import GRPOLab from './components/GRPOLab.vue'
import PPOClipLab from './components/PPOClipLab.vue'
import ReturnAdvantageLab from './components/ReturnAdvantageLab.vue'
import RLRolloutLab from './components/RLRolloutLab.vue'
import AgentShiftLab from './components/AgentShiftLab.vue'
import DistillationLab from './components/DistillationLab.vue'
import GeneralizationLab from './components/GeneralizationLab.vue'
import MLMLab from './components/MLMLab.vue'
import ICLSensitivityLab from './components/ICLSensitivityLab.vue'
import LoRALab from './components/LoRALab.vue'
import RAGGroundingLab from './components/RAGGroundingLab.vue'
import EvaluationThresholdLab from './components/EvaluationThresholdLab.vue'
import TokenFairnessLab from './components/TokenFairnessLab.vue'
import DiffusionNoiseLab from './components/DiffusionNoiseLab.vue'
import ActivationPatchingLab from './components/ActivationPatchingLab.vue'
import FlashAttentionLab from './components/FlashAttentionLab.vue'
import VLLMSchedulerLab from './components/VLLMSchedulerLab.vue'
import './styles/base.css'
import './styles/landing.css'
import './styles/components.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('HomePage', HomePage)
    app.component('ProgressTracker', ProgressTracker)
    app.component('KVCacheLab', KVCacheLab)
    app.component('AttentionLab', AttentionLab)
    app.component('MoERouterLab', MoERouterLab)
    app.component('PaperLibrary', PaperLibrary)
    app.component('ConceptCheck', ConceptCheck)
    app.component('TokenLab', TokenLab)
    app.component('TrainingLoopLab', TrainingLoopLab)
    app.component('TransformerArchitecture', TransformerArchitecture)
    app.component('LearningStagesDiagram', LearningStagesDiagram)
    app.component('K3SystemDiagram', K3SystemDiagram)
    app.component('SamplingLab', SamplingLab)
    app.component('RAGPipelineLab', RAGPipelineLab)
    app.component('VectorSimilarityLab', VectorSimilarityLab)
    app.component('BPELab', BPELab)
    app.component('ScalingLab', ScalingLab)
    app.component('QuantizationLab', QuantizationLab)
    app.component('PreferenceRLLab', PreferenceRLLab)
    app.component('ChapterReadings', ChapterReadings)
    app.component('RecurrentGradientLab', RecurrentGradientLab)
    app.component('ModernDecoderLab', ModernDecoderLab)
    app.component('TestTimeScalingLab', TestTimeScalingLab)
    app.component('GRPOLab', GRPOLab)
    app.component('PPOClipLab', PPOClipLab)
    app.component('ReturnAdvantageLab', ReturnAdvantageLab)
    app.component('RLRolloutLab', RLRolloutLab)
    app.component('AgentShiftLab', AgentShiftLab)
    app.component('DistillationLab', DistillationLab)
    app.component('GeneralizationLab', GeneralizationLab)
    app.component('MLMLab', MLMLab)
    app.component('ICLSensitivityLab', ICLSensitivityLab)
    app.component('LoRALab', LoRALab)
    app.component('RAGGroundingLab', RAGGroundingLab)
    app.component('EvaluationThresholdLab', EvaluationThresholdLab)
    app.component('TokenFairnessLab', TokenFairnessLab)
    app.component('DiffusionNoiseLab', DiffusionNoiseLab)
    app.component('ActivationPatchingLab', ActivationPatchingLab)
    app.component('FlashAttentionLab', FlashAttentionLab)
    app.component('VLLMSchedulerLab', VLLMSchedulerLab)
  }
}
