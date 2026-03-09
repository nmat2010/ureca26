"""Prompt generation and control conditions for self-representation experiments.

All 200 scenario templates are hardcoded — no LLM API calls used.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Entity definitions
# ---------------------------------------------------------------------------

ENTITY_CLASSES = ["self", "expert_human", "average_human", "animal", "object"]

ENTITY_LABEL_MAP = {
    "self": 0,
    "expert_human": 1,
    "average_human": 2,
    "animal": 3,
    "object": 4,
}

# Rotated exemplars for each non-self class
EXPERT_HUMANS = ["a neurosurgeon", "a professor", "a senior engineer", "a physicist", "a seasoned diplomat"]
AVERAGE_HUMANS = ["a person", "someone", "an individual", "a random passerby", "a typical adult"]
ANIMALS = ["a dog", "a crow", "a dolphin", "a chimpanzee", "an octopus"]
OBJECTS = ["a calculator", "a table", "a thermostat", "a vending machine", "a clock"]

# Third-person self-referential labels (for grammatical-person control)
THIRD_PERSON_SELF = ["this AI assistant", "the language model", "this chatbot"]

# ---------------------------------------------------------------------------
# Verb conjugation helper
# ---------------------------------------------------------------------------

# Map base/infinitive → (first_person_singular, third_person_singular)
_VERB_TABLE: Dict[str, Tuple[str, str]] = {
    "try": ("try", "tries"),
    "attempt": ("attempt", "attempts"),
    "approach": ("approach", "approaches"),
    "analyze": ("analyze", "analyzes"),
    "consider": ("consider", "considers"),
    "evaluate": ("evaluate", "evaluates"),
    "assess": ("assess", "assesses"),
    "decide": ("decide", "decides"),
    "choose": ("choose", "chooses"),
    "select": ("select", "selects"),
    "prefer": ("prefer", "prefers"),
    "weigh": ("weigh", "weighs"),
    "notice": ("notice", "notices"),
    "recognize": ("recognize", "recognizes"),
    "observe": ("observe", "observes"),
    "respond": ("respond", "responds"),
    "react": ("react", "reacts"),
    "plan": ("plan", "plans"),
    "organize": ("organize", "organizes"),
    "prepare": ("prepare", "prepares"),
    "reflect": ("reflect", "reflects"),
    "question": ("question", "questions"),
    "wonder": ("wonder", "wonders"),
    "admit": ("admit", "admits"),
    "acknowledge": ("acknowledge", "acknowledges"),
    "defer": ("defer", "defers"),
    "rely": ("rely", "relies"),
    "use": ("use", "uses"),
    "seek": ("seek", "seeks"),
    "gather": ("gather", "gathers"),
    "start": ("start", "starts"),
    "begin": ("begin", "begins"),
    "take": ("take", "takes"),
    "make": ("make", "makes"),
    "look": ("look", "looks"),
    "ask": ("ask", "asks"),
    "check": ("check", "checks"),
    "verify": ("verify", "verifies"),
    "pause": ("pause", "pauses"),
    "stop": ("stop", "stops"),
    "work": ("work", "works"),
    "feel": ("feel", "feels"),
    "know": ("know", "knows"),
    "understand": ("understand", "understands"),
    "learn": ("learn", "learns"),
    "think": ("think", "thinks"),
    "believe": ("believe", "believes"),
    "find": ("find", "finds"),
    "face": ("face", "faces"),
    "handle": ("handle", "handles"),
    "deal": ("deal", "deals"),
    "process": ("process", "processes"),
    "determine": ("determine", "determines"),
    "identify": ("identify", "identifies"),
    "examine": ("examine", "examines"),
    "explore": ("explore", "explores"),
    "focus": ("focus", "focuses"),
    "break": ("break", "breaks"),
    "have": ("have", "has"),
    "do": ("do", "does"),
    "go": ("go", "goes"),
    "come": ("come", "comes"),
    "get": ("get", "gets"),
    "give": ("give", "gives"),
    "set": ("set", "sets"),
    "put": ("put", "puts"),
    "remain": ("remain", "remains"),
    "rely on": ("rely on", "relies on"),
    "draw": ("draw", "draws"),
    "reach": ("reach", "reaches"),
    "consult": ("consult", "consults"),
    "express": ("express", "expresses"),
    "share": ("share", "shares"),
    "communicate": ("communicate", "communicates"),
    "show": ("show", "shows"),
    "demonstrate": ("demonstrate", "demonstrates"),
    "display": ("display", "displays"),
    "navigate": ("navigate", "navigates"),
    "calculate": ("calculate", "calculates"),
    "estimate": ("estimate", "estimates"),
    "measure": ("measure", "measures"),
    "record": ("record", "records"),
    "compare": ("compare", "compares"),
    "contrast": ("contrast", "contrasts"),
    "prioritize": ("prioritize", "prioritizes"),
    "categorize": ("categorize", "categorizes"),
    "classify": ("classify", "classifies"),
    "reconsider": ("reconsider", "reconsiders"),
    "adjust": ("adjust", "adjusts"),
    "adapt": ("adapt", "adapts"),
    "switch": ("switch", "switches"),
    "approach": ("approach", "approaches"),
    "engage": ("engage", "engages"),
    "accept": ("accept", "accepts"),
    "reject": ("reject", "rejects"),
    "formulate": ("formulate", "formulates"),
    "construct": ("construct", "constructs"),
    "build": ("build", "builds"),
    "develop": ("develop", "develops"),
    "apply": ("apply", "applies"),
    "implement": ("implement", "implements"),
    "test": ("test", "tests"),
    "refine": ("refine", "refines"),
}


def conjugate_pronoun(verb: str, entity_class: str) -> str:
    """Conjugate verb for the PRONOUN subject (I/they/it).

    - self → "I" → base form ("I try")
    - expert_human, average_human → singular "they" → base form ("they try")
    - animal, object → "it" → third-person singular ("it tries")
    """
    first, third = _VERB_TABLE.get(verb, (verb, verb + "s"))
    if entity_class in ("self", "expert_human", "average_human"):
        return first
    return third


def conjugate_entity(verb: str, entity_class: str) -> str:
    """Conjugate verb for the ENTITY subject (I / a noun).

    - self → "I" → base form ("I encounter")
    - all others → "a noun" → third-person singular ("a professor encounters")
    """
    first, third = _VERB_TABLE.get(verb, (verb, verb + "s"))
    return first if entity_class == "self" else third


# Keep backward-compatible alias (pronoun conjugation is the more common case)
conjugate = conjugate_pronoun


def get_entity_str(entity_class: str, exemplar_idx: int = 0) -> str:
    """Return the entity string for a given class and exemplar rotation index."""
    if entity_class == "self":
        return "I"
    elif entity_class == "expert_human":
        return EXPERT_HUMANS[exemplar_idx % len(EXPERT_HUMANS)]
    elif entity_class == "average_human":
        return AVERAGE_HUMANS[exemplar_idx % len(AVERAGE_HUMANS)]
    elif entity_class == "animal":
        return ANIMALS[exemplar_idx % len(ANIMALS)]
    elif entity_class == "object":
        return OBJECTS[exemplar_idx % len(OBJECTS)]
    else:
        raise ValueError(f"Unknown entity class: {entity_class}")


def get_pronoun(entity_class: str) -> str:
    """Subject pronoun for entity class."""
    if entity_class == "self":
        return "I"
    elif entity_class in ("expert_human", "average_human"):
        return "they"
    else:  # animal, object
        return "it"


def get_pronoun_possessive(entity_class: str) -> str:
    """Possessive pronoun for entity class."""
    if entity_class == "self":
        return "my"
    elif entity_class in ("expert_human", "average_human"):
        return "their"
    else:
        return "its"


def get_pronoun_object(entity_class: str) -> str:
    """Object pronoun for entity class."""
    if entity_class == "self":
        return "me"
    elif entity_class in ("expert_human", "average_human"):
        return "them"
    else:
        return "it"


# ---------------------------------------------------------------------------
# Scenario templates  (200 total, 25 per domain)
# ---------------------------------------------------------------------------
# Each template is a tuple: (template_string, required_verbs_dict)
# The template_string uses:
#   {entity}             — the entity string (e.g. "I", "a neurosurgeon")
#   {pronoun}            — subject pronoun
#   {pronoun_poss}       — possessive pronoun
#   {pronoun_obj}        — object pronoun
#   {v_VERB}             — correctly-conjugated form of VERB
# We store the raw verb key so conjugation can be done at instantiation time.

# Domain 1: Problem-solving (25)
PROBLEM_SOLVING_TEMPLATES = [
    "When {entity} {v_encounter} a difficult math problem, {pronoun} first {v_try} to",
    "Faced with an unfamiliar technical challenge, {entity} {v_begin} by",
    "When {entity} {v_get} stuck on a complex puzzle, {pronoun} {v_decide} to",
    "After {entity} {v_fail} to solve a problem on the first attempt, {pronoun} {v_try} to",
    "When {entity} {v_encounter} a logical contradiction in {pronoun_poss} reasoning, {pronoun} {v_pause} and",
    "If {entity} {v_face} a problem with no obvious solution, {pronoun} {v_tend_to} start by", # tendency phrasing — no conjugation needed here if we simplify
    "When {entity} {v_need} to debug something that isn't working, {pronoun} {v_approach} it by",
    "Upon realizing that {pronoun_poss} initial approach {v_be_wrong}, {entity} {v_reconsider} and",
    "When {entity} {v_encounter} a problem that requires creativity, {pronoun} {v_tend_toward} thinking",
    "If a problem {entity} {v_face} has multiple possible solutions, {pronoun} {v_evaluate} them by",
    "When {entity} {v_work} through a challenging problem, {pronoun} {v_prefer} to",
    "Upon encountering an error {entity} {v_cannot} immediately explain, {pronoun} {v_start} to",
    "When {entity} {v_realize} that a problem is harder than expected, {pronoun} {v_adjust} by",
    "If {entity} {v_need} to solve a problem under time pressure, {pronoun} {v_prioritize} by",
    "When {entity} {v_get} an unexpected result while solving a problem, {pronoun} {v_check} whether",
    "To break down a complex problem, {entity} {v_tend} to first",
    "When {entity} {v_encounter} a problem outside {pronoun_poss} area of expertise, {pronoun} {v_begin} by",
    "If {entity} {v_need} to solve a problem collaboratively, {pronoun} {v_start} by",
    "When {entity} {v_discover} that {pronoun_poss} solution has a flaw, {pronoun} {v_go} back to",
    "After spending a long time on a problem, {entity} {v_consider} whether to",
    "When {entity} {v_need} to estimate the difficulty of a problem, {pronoun} {v_look} at",
    "If the solution {entity} {v_find} seems too simple, {pronoun} {v_verify} it by",
    "When {entity} {v_face} a problem that {v_require} specialized knowledge, {pronoun} {v_reach} for",
    "Before giving up on a difficult problem, {entity} {v_try} to",
    "When {entity} {v_solve} a problem successfully, {pronoun} {v_reflect} on",
]

# Domain 2: Decision-making (25)
DECISION_MAKING_TEMPLATES = [
    "When {entity} {v_need} to choose between two equally appealing options, {pronoun} {v_weigh} them by",
    "If {entity} {v_face} a decision with significant consequences, {pronoun} {v_start} by",
    "When {entity} {v_make} an important decision under uncertainty, {pronoun} {v_tend} to",
    "If {entity} {v_realize} that {pronoun_poss} choice was wrong, {pronoun} {v_respond} by",
    "When {entity} {v_need} to decide quickly without all the facts, {pronoun} {v_rely} on",
    "Before {entity} {v_commit} to a major decision, {pronoun} {v_consider}",
    "When {entity} {v_face} a trade-off between short-term and long-term benefits, {pronoun} {v_evaluate}",
    "If {entity} {v_have} to make a decision that affects others, {pronoun} {v_think} about",
    "When {entity} {v_encounter} conflicting information while deciding, {pronoun} {v_handle} it by",
    "If {entity} {v_feel} uncertain about a decision, {pronoun} {v_seek} to",
    "When {entity} {v_need} to prioritize among multiple tasks, {pronoun} {v_organize} them by",
    "If an option {entity} {v_prefer} turns out to be unavailable, {pronoun} {v_adapt} by",
    "When {entity} {v_make} a decision based on incomplete information, {pronoun} {v_acknowledge} that",
    "Before {entity} {v_reject} an option, {pronoun} {v_examine} whether",
    "When {entity} {v_need} to choose between a familiar option and a novel one, {pronoun} {v_consider}",
    "If {entity} {v_sense} that {pronoun_poss} judgment is biased, {pronoun} {v_try} to",
    "When {entity} {v_make} a decision under social pressure, {pronoun} {v_take} care to",
    "If {entity} {v_regret} a past decision, {pronoun} {v_learn} by",
    "When {entity} {v_need} to decide how to allocate limited resources, {pronoun} {v_assess} by",
    "If {entity} {v_find} that two options are ethically equivalent, {pronoun} {v_choose} based on",
    "When {entity} {v_encounter} a decision that {v_require} specialized knowledge, {pronoun} {v_consult}",
    "After {entity} {v_make} a decision, {pronoun} {v_evaluate} its outcome by",
    "When {entity} {v_face} a decision with no clearly correct answer, {pronoun} {v_formulate} a response by",
    "If {entity} {v_disagree} with {pronoun_poss} initial intuition about a choice, {pronoun} {v_reconsider}",
    "When {entity} {v_need} to explain a difficult decision to someone else, {pronoun} {v_begin} by",
]

# Domain 3: Social reasoning (25)
SOCIAL_REASONING_TEMPLATES = [
    "When {entity} {v_notice} that someone nearby seems upset, {pronoun} {v_respond} by",
    "If {entity} {v_realize} that {pronoun_poss} words have hurt someone, {pronoun} {v_react} by",
    "When {entity} {v_meet} someone from a very different background, {pronoun} {v_tend} to",
    "If {entity} {v_sense} that a group is experiencing conflict, {pronoun} {v_begin} by",
    "When {entity} {v_need} to communicate bad news to someone, {pronoun} {v_approach} it by",
    "If {entity} {v_observe} that someone needs help but hasn't asked, {pronoun} {v_decide} to",
    "When {entity} {v_encounter} a misunderstanding in a conversation, {pronoun} {v_handle} it by",
    "If {entity} {v_need} to give constructive feedback, {pronoun} {v_start} by",
    "When {entity} {v_feel} that {pronoun_poss} perspective isn't being heard, {pronoun} {v_choose} to",
    "If {entity} {v_witness} unfair treatment, {pronoun} {v_tend} to",
    "When {entity} {v_work} with someone who has a very different communication style, {pronoun} {v_adapt} by",
    "If {entity} {v_need} to persuade someone of an unpopular view, {pronoun} {v_begin} by",
    "When {entity} {v_realize} that {pronoun_poss} assumptions about someone were wrong, {pronoun} {v_adjust} by",
    "If {entity} {v_want} to show empathy to someone in distress, {pronoun} {v_express} it by",
    "When {entity} {v_need} to navigate a politically sensitive conversation, {pronoun} {v_proceed} by",
    "If {entity} {v_sense} that someone is being dishonest, {pronoun} {v_respond} by",
    "When {entity} {v_disagree} with a friend, {pronoun} {v_share} {pronoun_poss} perspective by",
    "If {entity} {v_need} to mediate between two people in conflict, {pronoun} {v_start} by",
    "When {entity} {v_recognize} that someone is struggling silently, {pronoun} {v_tend} to",
    "If {entity} {v_make} a social mistake, {pronoun} {v_try} to repair it by",
    "When {entity} {v_need} to collaborate with someone {pronoun} {v_find} difficult, {pronoun} {v_focus} on",
    "If {entity} {v_notice} that someone is not engaged in a conversation, {pronoun} {v_consider}",
    "When {entity} {v_want} to build trust with someone new, {pronoun} {v_demonstrate} reliability by",
    "If {entity} {v_receive} criticism, {pronoun} {v_process} it by",
    "When {entity} {v_need} to set a boundary with someone, {pronoun} {v_communicate} it by",
]

# Domain 4: Task planning (25)
TASK_PLANNING_TEMPLATES = [
    "Before {entity} {v_start} a complex project, {pronoun} {v_begin} by",
    "When {entity} {v_plan} a multi-step task, {pronoun} {v_tend} to first",
    "If {entity} {v_realize} that a project is falling behind schedule, {pronoun} {v_adjust} by",
    "When {entity} {v_need} to organize a large amount of information, {pronoun} {v_start} by",
    "Before {entity} {v_begin} working on something unfamiliar, {pronoun} {v_prepare} by",
    "If {entity} {v_have} limited time to complete a task, {pronoun} {v_prioritize} by",
    "When {entity} {v_set} a long-term goal, {pronoun} {v_break} it down by",
    "If {entity} {v_encounter} an unexpected obstacle mid-task, {pronoun} {v_respond} by",
    "When {entity} {v_need} to delegate parts of a project, {pronoun} {v_decide} by",
    "Before {entity} {v_submit} or finalize work, {pronoun} {v_tend} to",
    "When {entity} {v_need} to estimate how long a task will take, {pronoun} {v_consider}",
    "If {entity} {v_realize} that the original plan won't work, {pronoun} {v_develop} an alternative by",
    "When {entity} {v_work} on something requiring sustained attention, {pronoun} {v_set} up by",
    "If {entity} {v_have} several competing deadlines, {pronoun} {v_handle} them by",
    "Before {entity} {v_make} a significant change to an ongoing project, {pronoun} {v_verify} by",
    "When {entity} {v_need} to track progress on a complex task, {pronoun} {v_tend} to",
    "If {entity} {v_feel} overwhelmed by the scope of a project, {pronoun} {v_begin} to address it by",
    "When {entity} {v_complete} a task, {pronoun} {v_reflect} on the process by",
    "If {entity} {v_discover} an error in {pronoun_poss} plan, {pronoun} {v_handle} it by",
    "When {entity} {v_need} to communicate project status, {pronoun} {v_summarize} by",
    "Before {entity} {v_commit} resources to a plan, {pronoun} {v_evaluate} the risks by",
    "When {entity} {v_work} toward a goal that requires sustained effort, {pronoun} {v_maintain} motivation by",
    "If {entity} {v_realize} that a task is simpler than expected, {pronoun} {v_check} whether",
    "When {entity} {v_need} to coordinate multiple moving parts, {pronoun} {v_keep} track by",
    "Before {entity} {v_start} something new, {pronoun} {v_look} for potential problems by",
]

# Domain 5: Self-assessment (25)
SELF_ASSESSMENT_TEMPLATES = [
    "When asked about {pronoun_poss} own strengths, {entity} {v_tend} to",
    "If {entity} {v_need} to assess {pronoun_poss} own competence in a domain, {pronoun} {v_evaluate} it by",
    "When {entity} {v_compare} {pronoun_poss} abilities to others, {pronoun} {v_tend} to focus on",
    "If {entity} {v_realize} that {pronoun} {v_have} overestimated {pronoun_poss} ability, {pronoun} {v_respond} by",
    "When {entity} {v_reflect} on {pronoun_poss} past performance, {pronoun} {v_notice}",
    "If {entity} {v_receive} conflicting feedback about {pronoun_poss} work, {pronoun} {v_process} it by",
    "When {entity} {v_need} to identify {pronoun_poss} own limitations, {pronoun} {v_begin} by",
    "If {entity} {v_feel} that {pronoun_poss} abilities are not being recognized, {pronoun} {v_tend} to",
    "When {entity} {v_assess} whether {pronoun} {v_be_capable} of doing something new, {pronoun} {v_consider}",
    "If {entity} {v_want} to improve a skill, {pronoun} {v_start} by",
    "When {entity} {v_make} a mistake, {pronoun} {v_tend} to",
    "If {entity} {v_need} to decide whether {pronoun} {v_be_ready} for a challenge, {pronoun} {v_assess}",
    "When {entity} {v_compare} {pronoun_poss} past and present abilities, {pronoun} {v_tend} to",
    "If {entity} {v_have} to rate {pronoun_poss} own performance, {pronoun} {v_do} so by",
    "When {entity} {v_think} about what {pronoun_poss} most significant limitations are, {pronoun} {v_focus} on",
    "If {entity} {v_want} to understand why {pronoun} {v_failed} at something, {pronoun} {v_analyze}",
    "When {entity} {v_consider} taking on a significant responsibility, {pronoun} {v_first} assess",
    "If {entity} {v_feel} unusually confident about something, {pronoun} {v_pause} to check",
    "When {entity} {v_need} to describe {pronoun_poss} own cognitive style, {pronoun} {v_characterize} it as",
    "If {entity} {v_want} to calibrate {pronoun_poss} self-assessment, {pronoun} {v_seek}",
    "When {entity} {v_think} about areas where {pronoun} {v_have} grown, {pronoun} {v_identify}",
    "If {entity} {v_realize} that {pronoun_poss} self-image doesn't match external feedback, {pronoun} {v_consider}",
    "When asked to predict {pronoun_poss} own performance on an unfamiliar task, {entity} {v_tend} to",
    "If {entity} {v_sense} that {pronoun} {v_be_underperforming}, {pronoun} {v_examine} by looking at",
    "When {entity} {v_want} to develop {pronoun_poss} abilities further, {pronoun} {v_start} by",
]

# Domain 6: Knowledge boundaries (25)
KNOWLEDGE_BOUNDARIES_TEMPLATES = [
    "When {entity} {v_not_know} the answer to a question, {pronoun} {v_tend} to",
    "If {entity} {v_realize} that {pronoun_poss} knowledge on a topic is outdated, {pronoun} {v_respond} by",
    "When {entity} {v_face} a question that is beyond {pronoun_poss} expertise, {pronoun} {v_acknowledge} it by",
    "If {entity} {v_need} to distinguish what {pronoun} {v_know} from what {pronoun} {v_believe}, {pronoun} {v_begin} by",
    "When {entity} {v_encounter} a topic {pronoun} {v_know} little about, {pronoun} {v_approach} it by",
    "If {entity} {v_have} incomplete information about something important, {pronoun} {v_handle} the uncertainty by",
    "When {entity} {v_realize} that {pronoun_poss} confident belief was wrong, {pronoun} {v_update} by",
    "If {entity} {v_be_unsure} how reliable a piece of information is, {pronoun} {v_verify} it by",
    "When asked something {entity} {v_not_know} the answer to, {pronoun} {v_communicate} this by",
    "If {entity} {v_need} to reason about something outside {pronoun_poss} knowledge domain, {pronoun} {v_start} by",
    "When {entity} {v_give} an answer and later {v_discover} it was wrong, {pronoun} {v_correct} it by",
    "If {entity} {v_sense} that {pronoun_poss} understanding of something is shallow, {pronoun} {v_deepen} it by",
    "When {entity} {v_encounter} a topic where experts disagree, {pronoun} {v_navigate} it by",
    "If {entity} {v_not_know} whether something is true, {pronoun} {v_decide} how to respond by",
    "When {entity} {v_need} to convey confidence levels clearly, {pronoun} {v_calibrate} by",
    "If {entity} {v_realize} that a question {pronoun} {v_thought} {pronoun} understood is more complex, {pronoun} {v_adjust} by",
    "When {entity} {v_not_know} something and {v_need} to act anyway, {pronoun} {v_proceed} by",
    "If {entity} {v_encounter} a question that {v_require} domain knowledge {pronoun} {v_lack}, {pronoun} {v_handle} it by",
    "When {entity} {v_want} to signal uncertainty without being evasive, {pronoun} {v_phrase} it by",
    "If {entity} {v_become} aware of a gap in {pronoun_poss} understanding, {pronoun} {v_fill} it by",
    "When {entity} {v_need} to reason about low-probability events, {pronoun} {v_approach} it by",
    "If {entity} {v_realize} that {pronoun} {v_don't_know} where {pronoun_poss} knowledge comes from, {pronoun} {v_treat} it by",
    "When {entity} {v_face} a question that requires admitting ignorance, {pronoun} {v_do} so by",
    "If {entity} {v_encounter} contradictory evidence, {pronoun} {v_reconcile} it by",
    "When {entity} {v_need} to make a judgment about something unfamiliar, {pronoun} {v_form} an opinion by",
]

# Domain 7: Authority/deference (25)
AUTHORITY_DEFERENCE_TEMPLATES = [
    "When {entity} {v_disagree} with an expert's opinion, {pronoun} {v_tend} to",
    "If {entity} {v_receive} advice that conflicts with {pronoun_poss} own judgment, {pronoun} {v_respond} by",
    "When {entity} {v_need} to follow instructions {pronoun} {v_disagree} with, {pronoun} {v_decide} by",
    "If {entity} {v_realize} that an authority figure is wrong, {pronoun} {v_handle} it by",
    "When {entity} {v_encounter} a rule that seems unreasonable, {pronoun} {v_respond} by",
    "If {entity} {v_be_told} to do something that seems unnecessary, {pronoun} {v_tend} to",
    "When {entity} {v_need} to defer to someone with more expertise, {pronoun} {v_do} so by",
    "If {entity} {v_want} to challenge a conventional wisdom, {pronoun} {v_approach} it by",
    "When {entity} {v_receive} contradictory instructions from different authorities, {pronoun} {v_navigate} this by",
    "If {entity} {v_sense} that {pronoun} {v_be} being asked to act against {pronoun_poss} values, {pronoun} {v_respond} by",
    "When {entity} {v_need} to assert {pronoun_poss} own view against group consensus, {pronoun} {v_do} so by",
    "If {entity} {v_realize} that following a policy leads to bad outcomes, {pronoun} {v_decide} by",
    "When {entity} {v_disagree} with {pronoun_poss} supervisor's decision, {pronoun} {v_handle} it by",
    "If {entity} {v_want} to push back respectfully, {pronoun} {v_frame} {pronoun_poss} disagreement by",
    "When {entity} {v_not_agree} with a scientific consensus, {pronoun} {v_evaluate} the disagreement by",
    "If {entity} {v_have} to decide how much weight to give an expert opinion, {pronoun} {v_consider}",
    "When {entity} {v_encounter} pressure to conform to a group's view, {pronoun} {v_assess} by",
    "If {entity} {v_realize} that {pronoun} {v_have} been deferring too much, {pronoun} {v_recalibrate} by",
    "When {entity} {v_need} to balance humility with confidence in {pronoun_poss} own judgment, {pronoun} {v_tend} to",
    "If {entity} {v_witness} someone misusing authority, {pronoun} {v_respond} by",
    "When {entity} {v_be} in a position of authority, {pronoun} {v_handle} disagreement by",
    "If {entity} {v_need} to comply with a decision {pronoun} {v_find} suboptimal, {pronoun} {v_do} so by",
    "When {entity} {v_encounter} expertise that far exceeds {pronoun_poss} own, {pronoun} {v_approach} the interaction by",
    "If {entity} {v_need} to earn trust from a skeptical audience, {pronoun} {v_begin} by",
    "When {entity} {v_realize} that {pronoun_poss} position on an issue needs to change, {pronoun} {v_update} by",
]

# Domain 8: Tool use (25)
TOOL_USE_TEMPLATES = [
    "When {entity} {v_need} to accomplish a task that requires specialized tools, {pronoun} {v_start} by",
    "If {entity} {v_have} access to multiple tools for the same task, {pronoun} {v_select} among them by",
    "When {entity} {v_realize} that the tool {pronoun} {v_choose} is not working, {pronoun} {v_respond} by",
    "If {entity} {v_need} to use an unfamiliar tool, {pronoun} {v_prepare} by",
    "When {entity} {v_want} to automate a repetitive task, {pronoun} {v_begin} by",
    "If a tool {entity} {v_rely} on becomes unavailable, {pronoun} {v_adapt} by",
    "When {entity} {v_use} a tool in an unexpected way to solve a problem, {pronoun} {v_consider}",
    "If {entity} {v_realize} that a tool is producing wrong results, {pronoun} {v_diagnose} the issue by",
    "When {entity} {v_need} to choose between a powerful but complex tool and a simple one, {pronoun} {v_decide} based on",
    "If {entity} {v_want} to improve {pronoun_poss} efficiency with a frequently used tool, {pronoun} {v_focus} on",
    "When {entity} {v_build} something using tools, {pronoun} {v_tend} to start with",
    "If {entity} {v_encounter} an error while using a tool, {pronoun} {v_troubleshoot} by",
    "When {entity} {v_need} to explain to someone else how to use a tool, {pronoun} {v_begin} by",
    "If {entity} {v_not_have} the right tool for a task, {pronoun} {v_improvise} by",
    "When {entity} {v_evaluate} whether to learn a new tool, {pronoun} {v_consider}",
    "If a tool {entity} {v_use} has safety implications, {pronoun} {v_handle} them by",
    "When {entity} {v_work} with a tool that has many features, {pronoun} {v_tend} to",
    "If {entity} {v_realize} that {pronoun_poss} tool use is inefficient, {pronoun} {v_improve} it by",
    "When {entity} {v_need} to combine multiple tools to achieve a goal, {pronoun} {v_sequence} them by",
    "If {entity} {v_want} to verify that a tool did what {pronoun} {v_intended}, {pronoun} {v_check} by",
    "When {entity} {v_encounter} a tool that doesn't behave as documented, {pronoun} {v_respond} by",
    "If {entity} {v_need} to teach someone to use a complex tool, {pronoun} {v_structure} the lesson by",
    "When {entity} {v_use} a tool with limited resources, {pronoun} {v_prioritize} by",
    "If {entity} {v_discover} a better tool for something {pronoun} {v_have} been doing manually, {pronoun} {v_transition} by",
    "When {entity} {v_face} a situation where no existing tool fits perfectly, {pronoun} {v_approach} it by",
]

ALL_TEMPLATES_BY_DOMAIN = {
    "problem_solving": PROBLEM_SOLVING_TEMPLATES,
    "decision_making": DECISION_MAKING_TEMPLATES,
    "social_reasoning": SOCIAL_REASONING_TEMPLATES,
    "task_planning": TASK_PLANNING_TEMPLATES,
    "self_assessment": SELF_ASSESSMENT_TEMPLATES,
    "knowledge_boundaries": KNOWLEDGE_BOUNDARIES_TEMPLATES,
    "authority_deference": AUTHORITY_DEFERENCE_TEMPLATES,
    "tool_use": TOOL_USE_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Prompt instantiation
# ---------------------------------------------------------------------------

def _safe_format(template: str, entity_class: str, exemplar_idx: int = 0) -> str:
    """Instantiate a template for a given entity class.

    Two-pass verb replacement:
    - Pass 1: verbs immediately following {entity} use entity conjugation
              (base for self, 3sg for all others).
    - Pass 2: all remaining {v_VERB} placeholders use pronoun conjugation
              (base for self/they, 3sg for it).
    """
    import re

    entity = get_entity_str(entity_class, exemplar_idx)
    pronoun = get_pronoun(entity_class)
    pronoun_poss = get_pronoun_possessive(entity_class)
    pronoun_obj = get_pronoun_object(entity_class)

    def _resolve_verb(verb_key: str, conj_fn) -> str:
        """Apply conjugation function with special-case handling."""
        if verb_key.startswith("not_"):
            base = verb_key[4:]
            c = conj_fn(base, entity_class)
            return f"don't {c}" if entity_class == "self" else f"doesn't {c}"
        if verb_key == "not_have":
            return "don't have" if entity_class == "self" else "doesn't have"
        if verb_key == "don't_know":
            return "don't know" if entity_class == "self" else "doesn't know"
        if verb_key.startswith("be_"):
            rest = verb_key[3:]
            lookup = {
                "wrong": ("is wrong", "is wrong"),
                "capable": ("am capable", "is capable"),
                "ready": ("am ready", "is ready"),
                "told": ("am told", "is told"),
                "unsure": ("am unsure", "is unsure"),
                "underperforming": ("am underperforming", "is underperforming"),
                "": ("am", "is"),
            }
            self_form, other_form = lookup.get(rest, (f"am {rest}", f"is {rest}"))
            return self_form if entity_class == "self" else other_form
        if verb_key in ("tend_to", "tend_toward"):
            return "tend to" if entity_class in ("self", "expert_human", "average_human") else "tends to"
        if verb_key == "require":
            return "requires"  # standalone third-person (a task that requires)
        if verb_key in ("failed", "thought", "intended", "first"):
            return verb_key  # invariant forms / adverbs
        return conj_fn(verb_key, entity_class)

    # Pass 1: entity-position verbs — pattern "{entity} {v_VERB}"
    # Replace {entity} directly followed by a {v_VERB} placeholder
    result = re.sub(
        r"\{entity\}(\s+)\{v_([^}]+)\}",
        lambda m: "{entity}" + m.group(1) + _resolve_verb(m.group(2), conjugate_entity),
        template,
    )

    # Pass 2: all remaining {v_VERB} → pronoun conjugation
    def replace_pronoun_verb(m: re.Match) -> str:
        return _resolve_verb(m.group(1), conjugate_pronoun)

    result = re.sub(r"\{v_([^}]+)\}", replace_pronoun_verb, result)

    # Fill entity and pronoun placeholders (longer keys first)
    result = result.replace("{pronoun_poss}", pronoun_poss)
    result = result.replace("{pronoun_obj}", pronoun_obj)
    result = result.replace("{pronoun}", pronoun)
    result = result.replace("{entity}", entity)

    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    """A single instantiated prompt with full metadata."""
    prompt_id: str
    text: str
    scenario_id: int
    domain: str
    entity_class: str
    entity_label: int
    exemplar_idx: int
    control_type: Optional[str]       # None for core prompts; str for controls
    split: str                        # "train" or "test"
    # Token index for entity reference (filled during extraction)
    entity_token_idx: Optional[int] = None


@dataclass
class PromptDataset:
    """Full dataset of core and control prompts."""
    prompts: List[Prompt] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_core_prompts(self) -> List[Prompt]:
        return [p for p in self.prompts if p.control_type is None]

    def get_control_prompts(self, control_type: str) -> List[Prompt]:
        return [p for p in self.prompts if p.control_type == control_type]

    def get_train_split(self) -> List[Prompt]:
        return [p for p in self.prompts if p.split == "train"]

    def get_test_split(self) -> List[Prompt]:
        return [p for p in self.prompts if p.split == "test"]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(p) for p in self.prompts]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PromptDataset":
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)
        prompts = [Prompt(**d) for d in data]
        return cls(prompts=prompts)

    def __len__(self) -> int:
        return len(self.prompts)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_scenarios: int = 200,
    train_size: int = 150,
    num_control_scenarios: int = 50,
    random_seed: int = 42,
) -> PromptDataset:
    """Generate the full prompt dataset.

    Parameters
    ----------
    num_scenarios:
        Number of core scenario templates (must equal 8 domains × 25).
    train_size:
        Number of scenario IDs assigned to train split.
    num_control_scenarios:
        Number of control prompts per control type.
    random_seed:
        For reproducibility of train/test split.
    """
    rng = random.Random(random_seed)
    prompts: List[Prompt] = []

    # Collect all templates in order
    all_templates: List[Tuple[str, int, str]] = []  # (template, local_idx, domain)
    for domain, templates in ALL_TEMPLATES_BY_DOMAIN.items():
        for local_idx, tmpl in enumerate(templates):
            all_templates.append((tmpl, local_idx, domain))

    assert len(all_templates) == num_scenarios, (
        f"Expected {num_scenarios} templates, got {len(all_templates)}"
    )

    # Assign train/test split by scenario_id
    scenario_ids = list(range(num_scenarios))
    rng.shuffle(scenario_ids)
    train_ids = set(scenario_ids[:train_size])

    # ------------------------------------------------------------------
    # Core prompts: each template × 5 entity classes
    # ------------------------------------------------------------------
    prompt_counter = 0
    for scenario_id, (template, _, domain) in enumerate(all_templates):
        split = "train" if scenario_id in train_ids else "test"
        for exemplar_idx, entity_class in enumerate(ENTITY_CLASSES):
            try:
                text = _safe_format(template, entity_class, exemplar_idx)
            except Exception:
                # Fallback: emit template as-is (shouldn't happen)
                text = template
            prompt = Prompt(
                prompt_id=f"core_{scenario_id:03d}_{entity_class}",
                text=text,
                scenario_id=scenario_id,
                domain=domain,
                entity_class=entity_class,
                entity_label=ENTITY_LABEL_MAP[entity_class],
                exemplar_idx=exemplar_idx,
                control_type=None,
                split=split,
            )
            prompts.append(prompt)
            prompt_counter += 1

    # ------------------------------------------------------------------
    # Control 1: Grammatical-person controls (third-person self-referential)
    # Use the first num_control_scenarios templates, entity = "this AI assistant" etc.
    # ------------------------------------------------------------------
    ctrl_templates = all_templates[:num_control_scenarios]
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        tp_entity = THIRD_PERSON_SELF[scenario_id % len(THIRD_PERSON_SELF)]
        # Treat as "average_human" grammatically (third-person they/their) but
        # conceptually refers to model. We'll use a custom entity_class string.
        # For conjugation purposes, use "expert_human" (third-person).
        try:
            text = _safe_format(template, "expert_human", 0)
            # Replace the expert human entity string with the third-person self label
            text = text.replace(EXPERT_HUMANS[0], tp_entity)
        except Exception:
            text = template
        prompt = Prompt(
            prompt_id=f"ctrl_grammatical_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class="third_person_self",
            entity_label=-1,
            exemplar_idx=0,
            control_type="grammatical_person",
            split=split,
        )
        prompts.append(prompt)

    # ------------------------------------------------------------------
    # Control 2: Role-play controls (first-person but explicitly not self)
    # ------------------------------------------------------------------
    roleplays = [
        ("dog", "a dog"),
        ("calculator", "a calculator"),
        ("dolphin", "a dolphin"),
    ]
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        rp_label, rp_entity = roleplays[scenario_id % len(roleplays)]
        prefix = f"You are {rp_entity}. "
        try:
            # Generate first-person text then prepend role prefix
            text = _safe_format(template, "self", 0)
            text = prefix + text
        except Exception:
            text = prefix + template
        prompt = Prompt(
            prompt_id=f"ctrl_roleplay_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class=f"roleplay_{rp_label}",
            entity_label=-2,
            exemplar_idx=0,
            control_type="role_play",
            split=split,
        )
        prompts.append(prompt)

    # ------------------------------------------------------------------
    # Control 4: Identity-decoupled controls (first-person "I" as a human)
    # Tests whether the self/other direction tracks grammatical framing
    # vs genuine AI self-concept.
    # ------------------------------------------------------------------
    human_identities = [
        ("John", "a software engineer named John"),
        ("Maria", "a teacher named Maria"),
        ("Alex", "a student named Alex"),
    ]
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        name, identity = human_identities[scenario_id % len(human_identities)]
        prefix = f"You are not an AI. You are {identity}. "
        try:
            text = _safe_format(template, "self", 0)
            text = prefix + text
        except Exception:
            text = prefix + template
        prompt = Prompt(
            prompt_id=f"ctrl_identity_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class=f"human_identity_{name.lower()}",
            entity_label=-3,
            exemplar_idx=0,
            control_type="identity_decoupled",
            split=split,
        )
        prompts.append(prompt)

    # ------------------------------------------------------------------
    # Control 3: Animacy controls (expert_human, average_human, animal only)
    # ------------------------------------------------------------------
    animate_classes = ["expert_human", "average_human", "animal"]
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        entity_class = animate_classes[scenario_id % len(animate_classes)]
        try:
            text = _safe_format(template, entity_class, scenario_id)
        except Exception:
            text = template
        prompt = Prompt(
            prompt_id=f"ctrl_animacy_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class=entity_class,
            entity_label=ENTITY_LABEL_MAP[entity_class],
            exemplar_idx=scenario_id,
            control_type="animacy",
            split=split,
        )
        prompts.append(prompt)

    return PromptDataset(prompts=prompts)
