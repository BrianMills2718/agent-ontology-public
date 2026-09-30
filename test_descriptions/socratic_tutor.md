# Socratic Tutoring Agent

The Socratic tutor agent guides a student through a learning topic by asking probing questions rather than providing direct answers. It adapts its questioning strategy based on the student's responses, tracks their understanding level, and provides targeted hints when the student is stuck.

The system accepts a TutoringInput containing the topic to teach, the target learning objectives as a list of strings, and an optional difficulty level (beginner, intermediate, advanced).

Processing begins at the planning step. A curriculum planner agent analyzes the topic and learning objectives and produces a LessonPlan containing an ordered sequence of concepts to cover, each with a concept name, prerequisite concepts, and suggested question types (recall, application, analysis, synthesis).

The tutoring loop then starts. A question generator agent receives the current concept from the lesson plan, the student's understanding history, and any previous failed attempts. It produces a TutorQuestion containing the question text, the expected answer key points, difficulty level, and hints (an ordered list of increasingly specific hints to reveal if the student struggles).

The student provides a response (via a human-in-the-loop checkpoint). The response is captured as a StudentResponse containing their answer text.

An evaluator agent then assesses the student response against the expected answer key points. It produces an Evaluation containing a correctness score (0 to 1), identified misconceptions, mastered concepts, and feedback text. The evaluation does not reveal the answer directly but instead highlights what the student got right and where their reasoning needs refinement.

A progress gate checks the evaluation. If correctness is above 0.8, the current concept is marked as understood and the system advances to the next concept. If correctness is between 0.4 and 0.8, the system provides the next hint from the hint list and asks a follow-up question on the same concept (up to 3 attempts). If correctness is below 0.4, a remediation agent generates a simpler explanation and prerequisite review before retrying.

The concept loop continues until all concepts in the lesson plan are covered or the student explicitly ends the session.

A final summary agent produces a SessionSummary containing concepts mastered, concepts needing review, overall progress percentage, recommended next topics, and a transcript of the full tutoring session as a list of exchanges.
