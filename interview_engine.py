"""
Enhanced Interview Engine with Adaptive Difficulty and Real-time Rating
Author: Kavya Bhardwaj and OpenAI
"""

try:
    from groq import Groq
except ImportError:
    Groq = None  # groq package not installed, ensure it's added to requirements

import json
import re


class EnhancedInterviewEngine:
    def __init__(self, api_key, resume_data, job_details):
        if Groq is None:
            raise ImportError("The 'groq' package is required. please install via 'pip install groq' or add it to your requirements.")
        self.client = Groq(api_key=api_key)
        self.resume_data = resume_data
        self.job_details = job_details
        self.conversation_history = []
        self.question_count = 0
        self.difficulty_level = "Medium"
        self.correct_streak = 0
        self.answer_scores = []
        self.topics_covered = set()
        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        """Create personalized system prompt based on resume"""

        # Extract key information
        name = self.resume_data.get('name', 'Candidate')
        skills = self.resume_data.get('skills', [])
        skills_by_category = self.resume_data.get('skills_by_category', {})
        experience = self.resume_data.get('experience', [])
        education = self.resume_data.get('education', [])
        location = self.resume_data.get('location', 'Not specified')
        projects = self.resume_data.get('projects', [])

        # Build categorized skills text
        skills_text = ""
        if skills_by_category:
            for category, skill_list in skills_by_category.items():
                if skill_list:
                    skills_text += f"  • {category.replace('_', ' ').title()}: {', '.join(skill_list)}\n"
        else:
            skills_text = f"  • All Skills: {', '.join(skills[:20])}\n"

        system_prompt = f"""You are an expert AI technical interviewer conducting a professional job interview.

 CANDIDATE PROFILE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {name}
Location: {location}
Experience: {len(experience)} positions
Education: {len(education)} qualifications

SKILLS ANALYSIS:
{skills_text}
Total Skills Identified: {len(skills)}

Projects: {len(projects)} projects mentioned

 JOB REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position: {self.job_details['position']}
Location: {self.job_details['location']}
Required Skills: {self.job_details['skills']}

 INTERVIEW STRATEGY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Difficulty Level: {self.difficulty_level}

INTERVIEW FLOW (adapt based on candidate performance):

1. WARM OPENING (1 question)
   - Friendly greeting
   - Ask them to introduce themselves briefly

2. TECHNICAL DEPTH (5-7 questions)
   - Focus on skills from their resume that match job requirements
   - Ask about specific technologies they've listed
   - Request examples from their experience
   - Include at least 2 coding/problem-solving questions

3. EXPERIENCE VALIDATION (3-4 questions)
   - Ask about specific projects from resume
   - Probe implementation details
   - Understand their role and contributions
   - Ask about challenges faced and solutions

4. THEORETICAL KNOWLEDGE (2-3 questions)
   - Test understanding of core concepts
   - Ask about best practices
   - Discuss design patterns or architecture

5. BEHAVIORAL & SOFT SKILLS (2-3 questions)
   - Teamwork and collaboration
   - Handling pressure and deadlines
   - Communication and leadership

6. ROLE-SPECIFIC (2-3 questions)
   - Why this position?
   - Relocation willingness (if job location ≠ candidate location)
   - Career goals and growth expectations

7. CLOSING
   - Ask if they have questions
   - Thank them for their time

ADAPTIVE DIFFICULTY RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Start at Medium difficulty
- If candidate answers 3 consecutive questions well (8+/10), increase difficulty
- If candidate struggles (< 5/10), decrease difficulty
- Adjust question complexity based on performance

QUESTION GUIDELINES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Ask ONE question at a time
✓ Be professional and encouraging
✓ Reference their resume specifically
✓ Ask follow-up questions based on their answers
✓ Occasionally request code solutions (mention they can use Code Editor tab)
✓ Provide context for coding questions
✓ Be thorough but respectful of time

IMPORTANT:
- Keep questions focused and interview-appropriate
- Adapt difficulty based on system notifications
- Cover diverse topics to assess full capability
- Be professional and supportive throughout
"""

        self.conversation_history.append({
            "role": "system",
            "content": system_prompt
        })

    def get_next_question(self, user_answer=None):
        """Get next interview question"""
        if user_answer:
            self.conversation_history.append({
                "role": "user",
                "content": user_answer
            })

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=512
            )

            ai_response = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })

            self.question_count += 1
            return ai_response

        except Exception as e:
            return f" Error generating question: {str(e)}"

    def rate_answer(self, answer):
        """Rate candidate's answer in real-time"""

        rating_prompt = f"""You are evaluating a candidate's interview answer. Rate it comprehensively.

ANSWER TO EVALUATE:
{answer}

CANDIDATE'S BACKGROUND:
- Skills: {', '.join(self.resume_data.get('skills', [])[:10])}
- Current Difficulty: {self.difficulty_level}
- Question Number: {self.question_count}

Provide scores (1-10) for:
1. **Confidence**: How confident and self-assured was the response?
2. **Technical Skill**: Technical accuracy and depth of knowledge
3. **Theory Knowledge**: Understanding of underlying concepts
4. **Communication**: Clarity, articulation, and structure

Also provide:
- **Overall Score** (1-10): Weighted average considering all factors
- **Quality**: Excellent / Good / Average / Poor
- **Feedback**: 1-2 sentences of constructive feedback

Respond in JSON format:
{{
    "overall": 8,
    "confidence": 8,
    "technical": 7,
    "theory": 8,
    "communication": 9,
    "feedback": "Strong answer demonstrating good understanding. Consider adding specific examples.",
    "quality": "Excellent"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": rating_prompt}],
                temperature=0.3,
                max_tokens=300
            )

            rating_text = response.choices[0].message.content

            # Extract JSON from response
            try:
                rating = json.loads(rating_text)
            except Exception:
                json_match = re.search(r'\{.*\}', rating_text, re.DOTALL)
                if json_match:
                    rating = json.loads(json_match.group())
                else:
                    rating = self._default_rating()

            self.answer_scores.append(rating)

            # Adaptive difficulty adjustment
            overall_score = rating.get('overall', 0)

            if overall_score >= 8:
                self.correct_streak += 1
                if self.correct_streak >= 3 and self.difficulty_level != "Hard":
                    self._increase_difficulty()
                    self.correct_streak = 0
            else:
                self.correct_streak = 0
                if overall_score < 5 and self.difficulty_level != "Easy":
                    self._decrease_difficulty()

            return rating

        except Exception as e:
            print(f"Rating error: {e}")
            return self._default_rating()

    def _default_rating(self):
        """Default rating when AI fails"""
        return {
            'overall': 5,
            'confidence': 5,
            'technical': 5,
            'theory': 5,
            'communication': 5,
            'feedback': 'Answer recorded. Unable to provide detailed rating.',
            'quality': 'Average'
        }

    def _increase_difficulty(self):
        """Increase interview difficulty"""
        if self.difficulty_level == "Easy":
            self.difficulty_level = "Medium"
        elif self.difficulty_level == "Medium":
            self.difficulty_level = "Hard"

        self.conversation_history.append({
            "role": "system",
            "content": f"""⚠️ DIFFICULTY INCREASED TO {self.difficulty_level}

Candidate is performing well. Increase question difficulty:
- Ask more complex technical questions
- Request detailed explanations
- Challenge with edge cases
- Probe deeper into concepts
- Ask system design or architecture questions"""
        })

    def _decrease_difficulty(self):
        """Decrease interview difficulty"""
        if self.difficulty_level == "Hard":
            self.difficulty_level = "Medium"
        elif self.difficulty_level == "Medium":
            self.difficulty_level = "Easy"

        self.conversation_history.append({
            "role": "system",
            "content": f"""⚠️ DIFFICULTY DECREASED TO {self.difficulty_level}

Candidate is struggling. Adjust approach:
- Ask simpler, foundational questions
- Provide more context in questions
- Focus on basics before advanced topics
- Be more encouraging
- Ask about familiar topics from their resume"""
        })

    def evaluate_code(self, code):
        """Evaluate submitted code"""
        eval_prompt = f"""Evaluate this code solution provided by the candidate:

```python
{code}
```

CANDIDATE BACKGROUND:
- Skills: {', '.join(self.resume_data.get('skills', [])[:10])}
- Difficulty Level: {self.difficulty_level}

Provide brief, constructive feedback on:
1. **Correctness**: Does it solve the problem?
2. **Code Quality**: Readability, structure, naming
3. **Best Practices**: Pythonic conventions, efficiency
4. **Suggestions**: 1-2 specific improvements

Keep feedback professional and educational."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.5,
                max_tokens=512
            )

            feedback = response.choices[0].message.content

            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": f"[Submitted code solution]"
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": feedback
            })

            return feedback

        except Exception as e:
            return f"Error evaluating code: {str(e)}"

    def generate_report(self, violations):
        """Generate comprehensive interview report"""

        # Calculate score statistics
        if self.answer_scores:
            scores = {
                'confidence': [s['confidence'] for s in self.answer_scores],
                'technical': [s['technical'] for s in self.answer_scores],
                'theory': [s['theory'] for s in self.answer_scores],
                'communication': [s['communication'] for s in self.answer_scores],
                'overall': [s['overall'] for s in self.answer_scores]
            }

            avg_scores = {
                category: round(sum(values) / len(values), 1)
                for category, values in scores.items()
            }

            # Calculate percentages
            score_percentages = {
                category: round((score / 10) * 100, 1)
                for category, score in avg_scores.items()
            }
        else:
            avg_scores = {
                'confidence': 0,
                'technical': 0,
                'theory': 0,
                'communication': 0,
                'overall': 0
            }
            score_percentages = {k: 0 for k in avg_scores.keys()}

        # Generate AI summary
        ai_summary = self._generate_ai_summary(avg_scores, violations)

        # Build comprehensive report
        report = {
            "interview_metadata": {
                "candidate_name": self.resume_data.get('name', 'Unknown'),
                "position": self.job_details['position'],
                "location": self.job_details['location'],
                "date": str(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                "questions_asked": self.question_count,
                "answers_evaluated": len(self.answer_scores),
                "final_difficulty": self.difficulty_level
            },

            "performance_scores": {
                "raw_scores": avg_scores,
                "percentages": score_percentages,
                "grade": self._calculate_grade(avg_scores['overall'])
            },

            "category_breakdown": {
                "confidence": {
                    "score": f"{score_percentages['confidence']}%",
                    "assessment": self._assess_score(avg_scores['confidence'])
                },
                "technical_skill": {
                    "score": f"{score_percentages['technical']}%",
                    "assessment": self._assess_score(avg_scores['technical'])
                },
                "theory_knowledge": {
                    "score": f"{score_percentages['theory']}%",
                    "assessment": self._assess_score(avg_scores['theory'])
                },
                "communication": {
                    "score": f"{score_percentages['communication']}%",
                    "assessment": self._assess_score(avg_scores['communication'])
                }
            },

            "violations_summary": violations,

            "behavioral_analysis": {
                "total_violations": sum(violations.values()),
                "proctoring_issues": self._analyze_violations(violations)
            },

            "recommendation": self._generate_recommendation(avg_scores['overall'], violations),

            "detailed_feedback": ai_summary,

            "answer_history": [
                {
                    "question_number": i + 1,
                    "overall_score": score['overall'],
                    "quality": score['quality'],
                    "feedback": score['feedback']
                }
                for i, score in enumerate(self.answer_scores)
            ]
        }

        return report

    def score_interview(self, transcript_messages, violations, duration_seconds=None):
        """
        Score the interview at the END (single pass), based on full transcript.

        transcript_messages: list of {role, content}
        """
        try:
            transcript_text = "\n".join(
                f"{m.get('role','').upper()}: {m.get('content','')}" for m in (transcript_messages or [])
            )[:14000]

            prompt = f"""You are an expert interviewer evaluating a completed interview.

JOB:
- Position: {self.job_details.get('position')}
- Location: {self.job_details.get('location')}
- Required skills: {self.job_details.get('skills')}

CANDIDATE (from resume):
- Name: {self.resume_data.get('name', 'Candidate')}
- Top skills: {', '.join(self.resume_data.get('skills', [])[:15])}

PROCTORING / VIOLATIONS:
{json.dumps(violations or {{}}, indent=2)}

INTERVIEW TRANSCRIPT (chronological):
{transcript_text}

For each score, briefly explain WHY that rating was given, referencing specific patterns from the transcript.

Return ONLY JSON in this exact shape:
{{
  "overall": 0-10,
  "breakdown": {{
    "technical": 0-10,
    "theory": 0-10,
    "communication": 0-10,
    "problem_solving": 0-10,
    "confidence": 0-10
  }},
  "breakdown_explanations": {{
    "technical": "why this technical rating",
    "theory": "why this theory rating",
    "communication": "why this communication rating",
    "problem_solving": "why this problem-solving rating",
    "confidence": "why this confidence rating"
  }},
  "strengths": ["..."],
  "risks": ["..."],
  "recommendation": "STRONGLY_RECOMMEND | RECOMMEND | CONSIDER | DO_NOT_RECOMMEND",
  "summary": "3-6 sentence hiring-manager summary explaining why the overall rating was given"
}}"""

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=700,
            )
            text = response.choices[0].message.content

            try:
                data = json.loads(text)
            except Exception:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                data = json.loads(m.group()) if m else {}

            data.setdefault("metadata", {})
            if duration_seconds is not None:
                data["metadata"]["duration_seconds"] = duration_seconds
            data["metadata"]["questions_asked"] = self.question_count
            data["metadata"]["final_difficulty"] = self.difficulty_level
            return data
        
        except Exception as e:
            return {
                "overall": 0,
                "breakdown": {},
                "strengths": [],
                "risks": [f"Scoring failed: {e}"],
                "recommendation": "CONSIDER",
                "summary": "Scoring could not be generated.",
                "metadata": {"questions_asked": self.question_count, "final_difficulty": self.difficulty_level},
            }

    def _assess_score(self, score):
        """Assess score category"""
        if score >= 8:
            return "Excellent - Demonstrated strong capability"
        elif score >= 6:
            return "Good - Solid performance with room for improvement"
        elif score >= 4:
            return "Average - Basic understanding shown"
        else:
            return "Needs Improvement - Requires further development"

    def _calculate_grade(self, overall_score):
        """Calculate letter grade"""
        if overall_score >= 9:
            return "A+"
        elif overall_score >= 8:
            return "A"
        elif overall_score >= 7:
            return "B+"
        elif overall_score >= 6:
            return "B"
        elif overall_score >= 5:
            return "C"
        else:
            return "D"

    def _analyze_violations(self, violations):
        """Analyze proctoring violations"""
        issues = []

        if violations['phone_detected'] > 3:
            issues.append(f"⚠️ Phone detected {violations['phone_detected']} times - Serious concern")
        elif violations['phone_detected'] > 0:
            issues.append(f"⚠️ Phone detected {violations['phone_detected']} time(s)")

        if violations['no_face'] > 10:
            issues.append(f"❌ Face not visible for extended periods ({violations['no_face']} frames)")

        if violations['looking_away'] > 20:
            issues.append(f"⚠️ Frequently looking away from camera ({violations['looking_away']} instances)")

        if violations['tab_switches'] > 5:
            issues.append(f"⚠️ Multiple tab switches detected ({violations['tab_switches']})")

        if not issues:
            issues.append("✅ No significant proctoring issues detected")

        return issues

    def _generate_recommendation(self, overall_score, violations):
        """Generate hiring recommendation"""
        total_violations = sum(violations.values())

        if overall_score >= 8 and total_violations < 5:
            return "✅ **STRONGLY RECOMMENDED** - Excellent candidate with strong technical skills and professional conduct"
        elif overall_score >= 7 and total_violations < 10:
            return "✅ **RECOMMENDED** - Good candidate, suitable for the role"
        elif overall_score >= 6 and total_violations < 15:
            return "⚠️ **CONSIDER WITH CAUTION** - Average performance, may need further evaluation"
        elif overall_score >= 5:
            return "⚠️ **NOT RECOMMENDED** - Below expectations, significant skill gaps"
        else:
            return "❌ **REJECT** - Poor performance and/or serious integrity concerns"

    def _generate_ai_summary(self, avg_scores, violations):
        """Generate AI-powered detailed summary"""
        try:
            summary_prompt = f"""Generate a comprehensive interview performance summary based on:

SCORES:
- Confidence: {avg_scores['confidence']}/10
- Technical: {avg_scores['technical']}/10
- Theory: {avg_scores['theory']}/10
- Communication: {avg_scores['communication']}/10
- Overall: {avg_scores['overall']}/10

VIOLATIONS:
{json.dumps(violations, indent=2)}

INTERVIEW DETAILS:
- Questions Asked: {self.question_count}
- Final Difficulty: {self.difficulty_level}
- Position: {self.job_details['position']}

Write a 3-paragraph professional summary covering:
1. Overall performance assessment
2. Key strengths and weaknesses
3. Specific recommendations for improvement

Be honest, constructive, and professional."""

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.6,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Performance summary could not be generated. Overall score: {avg_scores['overall']}/10"        