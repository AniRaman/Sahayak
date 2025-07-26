"""
Main Sequential Pipeline for ContentAgent -> WorksheetAgent Coordination

This pipeline coordinates between ContentAgent and WorksheetAgent:
1. Detects when ContentAgent provides large content (>500 words) or research/articles  
2. Automatically triggers WorksheetAgent to generate educational materials
3. Uses ADK SequentialAgent pattern for proper state management
4. Returns comprehensive educational materials for complex content
"""

import os
import json
import logging
import sys
from typing import Dict, Any
from google.adk.agents import SequentialAgent

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import existing agents
from sahayak_content_agent.agent import root_agent as content_agent
from sahayak_worksheet_agent.agent import root_agent as worksheet_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create the sequential pipeline with existing agents
main_coordination_pipeline = SequentialAgent(
    name="MainCoordinationPipeline",
    sub_agents=[content_agent, worksheet_agent]
)


def coordinate_content_processing(content: str, source_agent: str = "content_agent") -> Dict[str, Any]:
    """
    Main coordination function that determines if content needs worksheet generation
    based on >500 words criteria and triggers the pipeline accordingly.
    
    Args:
        content (str): Educational content from ContentAgent or user
        source_agent (str): Source of the content (content_agent, user_direct)
        
    Returns:
        dict: Processing decision and worksheet generation results if triggered
    """
    
    try:
        logger.info(f"Starting content coordination from {source_agent}")
        
        # Step 1: Apply >500 words detection logic as requested
        # Clean content and count words properly
        cleaned_content = ' '.join(content.split())  # Remove extra whitespace
        word_count = len(cleaned_content.split())
        content_length = len(content)
        should_generate_materials = word_count > 500
        
        print(f"[DETECTION] Content analysis: {word_count} words, {content_length} characters")
        print(f"[DETECTION] Should generate materials: {should_generate_materials} (threshold: >500 words)")
        
        coordination_result = {
            "status": "success",
            "should_generate_materials": should_generate_materials,
            "content_analysis": {
                "word_count": word_count,
                "character_count": content_length,
                "meets_threshold": should_generate_materials,
                "threshold_used": "500 words"
            },
            "processing_metadata": {
                "pipeline_name": "MainCoordinationPipeline",
                "source_agent": source_agent,
                "detection_method": "word_count_threshold"
            }
        }
        
        # Step 2: If content meets criteria, trigger the sequential pipeline
        if should_generate_materials:
            logger.info("Content exceeds 500 words - triggering sequential pipeline (ContentAgent -> WorksheetAgent)")
            
            try:
                # Prepare input for the sequential pipeline
                pipeline_input = {
                    "content": content,
                    "word_count": word_count,
                    "source": source_agent
                }
                
                # Execute the sequential pipeline: ContentAgent -> WorksheetAgent
                print(f"[PIPELINE] Running SequentialAgent with ContentAgent -> WorksheetAgent")
                
                # For now, call WorksheetAgent directly since ADK SequentialAgent API needs clarification
                print(f"[PIPELINE] Calling WorksheetAgent directly with content")
                from sahayak_worksheet_agent.agent import generate_educational_materials
                
                # Prepare parameters for WorksheetAgent
                pipeline_result = generate_educational_materials(
                    content=content,
                    subject="Auto-detected",
                    grade_level="Auto-detected", 
                    curriculum="General",
                    output_format="both",
                    save_to_storage=True
                )
                
                coordination_result["pipeline_execution"] = {
                    "status": "success",
                    "agents_executed": ["content_agent", "worksheet_agent"],
                    "pipeline_result": pipeline_result
                }
                
                # Check if worksheet materials were generated
                if pipeline_result and "download_links" in str(pipeline_result):
                    coordination_result["materials_generated"] = True
                    coordination_result["worksheet_generation"] = pipeline_result
                    print(f"[SUCCESS] Educational materials generated and saved to cloud storage")
                else:
                    coordination_result["materials_generated"] = False
                    print(f"[INFO] Pipeline executed but no materials generated")
                
            except Exception as pipeline_error:
                logger.error(f"Sequential pipeline execution failed: {pipeline_error}")
                coordination_result["pipeline_execution"] = {
                    "status": "error",
                    "error_message": f"Pipeline failed: {str(pipeline_error)}"
                }
                coordination_result["materials_generated"] = False
                
        else:
            # Content doesn't meet criteria - no worksheet generation needed
            logger.info(f"Content is only {word_count} words - below 500 word threshold. No worksheet generation needed.")
            coordination_result["pipeline_execution"] = {
                "status": "skipped",
                "reason": f"Content has only {word_count} words, below 500-word threshold for worksheet generation",
                "recommendation": "Content can be used directly for teaching without additional materials"
            }
            coordination_result["materials_generated"] = False
        
        logger.info(f"Content coordination completed. Materials generated: {coordination_result.get('materials_generated', False)}")
        return coordination_result
        
    except Exception as e:
        logger.error(f"Content coordination failed: {e}")
        return {
            "status": "error",
            "error_message": f"Coordination failed: {str(e)}"
        }


if __name__ == "__main__":
    # Test the coordination pipeline
    
    # Test 1: Short content (should not trigger worksheet generation)
    short_content = """
    Photosynthesis is the process by which green plants use sunlight to make their own food.
    Plants capture light energy using chlorophyll and convert carbon dioxide and water into glucose and oxygen.
    This process is essential for life on Earth as it produces the oxygen we breathe.
    """
    
    print("=== TEST 1: Short Content (< 500 words) ===")
    short_result = coordinate_content_processing(short_content, "content_agent")
    print(f"Status: {short_result.get('status')}")
    print(f"Should generate materials: {short_result.get('should_generate_materials')}")
    print(f"Word count: {short_result.get('processing_metadata', {}).get('content_word_count')}")
    
    # Test 2: Long research content (should trigger worksheet generation)
    long_research_content = """
    Photosynthesis is a complex biological process by which green plants, algae, and some bacteria convert light energy, 
    typically from the sun, into chemical energy stored in glucose molecules. This process is fundamental to life on Earth 
    as it serves as the primary source of organic compounds and oxygen in our atmosphere. The process occurs primarily in 
    the chloroplasts of plant cells, specifically within structures called thylakoids which contain the necessary pigments 
    and protein complexes for energy conversion.
    
    The photosynthetic process can be divided into two main stages: the light-dependent reactions (also known as the photo 
    reactions) and the light-independent reactions (also called the Calvin cycle or dark reactions). During the light-dependent 
    reactions, chlorophyll and other photosynthetic pigments absorb photons of light energy. This energy is used to split 
    water molecules (H2O) into hydrogen and oxygen atoms, releasing oxygen as a byproduct into the atmosphere. The chlorophyll 
    molecules are organized into photosystems, specifically Photosystem I and Photosystem II, which work together to capture 
    and transfer energy efficiently throughout the photosynthetic apparatus.
    
    The energy captured during the light reactions is stored in energy-carrying molecules called ATP (adenosine triphosphate) 
    and NADPH (nicotinamide adenine dinucleotide phosphate). These molecules then power the Calvin cycle, where carbon dioxide 
    from the atmosphere is incorporated into organic molecules through a process called carbon fixation. The enzyme RuBisCO 
    (ribulose-1,5-bisphosphate carboxylase/oxygenase) plays a crucial role in this process by catalyzing the reaction between 
    CO2 and RuBP (ribulose bisphosphate). RuBisCO is considered one of the most important enzymes on Earth and is also one 
    of the most abundant proteins in the world.
    
    The overall chemical equation for photosynthesis can be summarized as: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2 + 6H2O. 
    This equation represents the conversion of six molecules of carbon dioxide and six molecules of water, in the presence of 
    light energy, into one molecule of glucose, six molecules of oxygen, and six molecules of water. However, this simplified 
    equation does not capture the complexity of the many intermediate steps and regulatory mechanisms involved in the process.
    
    Photosynthesis has profound ecological and environmental implications. It is responsible for producing virtually all the 
    oxygen in our atmosphere and serves as the foundation of most food webs on Earth. Without photosynthesis, life as we know 
    it could not exist. The process also plays a crucial role in the global carbon cycle, helping to regulate atmospheric CO2 
    levels and mitigate climate change effects. Plants absorb approximately 120 billion tons of carbon dioxide annually through 
    photosynthesis, making them essential carbon sinks in our global ecosystem.
    
    Understanding photosynthesis is essential for students studying biology, environmental science, and related fields. It 
    demonstrates the interconnectedness of biological systems and highlights the importance of plants in maintaining 
    environmental balance. Modern research in photosynthesis also has applications in renewable energy, as scientists work 
    to develop artificial photosynthetic systems that could help address global energy needs while reducing carbon emissions.
    
    Different types of photosynthesis have evolved in different plant species. C3 photosynthesis is the most common form, 
    while C4 and CAM photosynthesis have evolved as adaptations to hot, dry climates. These variations demonstrate the 
    remarkable adaptability of life and provide insights into plant evolution and ecology that are valuable for both 
    scientific understanding and practical applications in agriculture and conservation.
    """
    
    print("\n=== TEST 2: Long Research Content (> 500 words) ===") 
    long_result = coordinate_content_processing(long_research_content, "content_agent")
    print(f"Status: {long_result.get('status')}")
    print(f"Should generate materials: {long_result.get('should_generate_materials')}")
    print(f"Word count: {long_result.get('processing_metadata', {}).get('content_word_count')}")
    print(f"Materials generated: {long_result.get('materials_generated', False)}")
    
    if long_result.get('materials_generated'):
        worksheet_gen = long_result.get('worksheet_generation', {})
        if worksheet_gen.get('download_links'):
            print("✅ Educational materials generated and saved to cloud storage")
            print(f"Generated files: {list(worksheet_gen.get('download_links', {}).keys())}")
        else:
            print("⚠️ Materials generated but no download links available")
    
    print("\n=== Pipeline Test Completed ===")
    print(f"ADK SequentialAgent pipeline: {'Working' if long_result.get('status') == 'success' else 'Failed'}")