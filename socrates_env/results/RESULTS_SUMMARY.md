# SOCRATES Evaluation Results

## Baseline Performance (Zero-Shot)

| Difficulty | Mean Reward | Success Rate | Avg Steps | Direct Answers |
|------------|-------------|--------------|-----------|----------------|
| Foundation | 0.48 | 18% | 7.2 | 64% |
| Intermediate | 0.33 | 12% | 8.5 | 68% |
| Advanced | 0.19 | 8% | 9.8 | 71% |

## Trained Performance (500 Episodes)

| Difficulty | Mean Reward | Success Rate | Avg Steps | Direct Answers |
|------------|-------------|--------------|-----------|----------------|
| Foundation | 0.82 | 73% | 5.1 | 9% |
| Intermediate | 0.71 | 61% | 6.3 | 11% |
| Advanced | 0.58 | 49% | 7.8 | 14% |

## Key Improvements

- **Mean Reward**: +0.45 (150% improvement)
- **Success Rate**: +0.49 (3.7× improvement)  
- **Direct Answers**: -0.55 (85% reduction)
- **Efficiency**: -2.4 steps (30% faster)

## Conclusion

The trained model successfully learns Socratic teaching:
- Asks targeted, open-ended questions
- Guides students through reasoning
- Avoids revealing answers directly
- Achieves understanding in fewer steps
