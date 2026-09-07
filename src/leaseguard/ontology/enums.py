"""Stable vocabulary for LeaseGuard schema version 1."""

from enum import StrEnum


class DocumentType(StrEnum):
    """Supported document types."""

    OFFICE_LEASE = "office_lease"
    RETAIL_LEASE = "retail_lease"
    AMENDMENT = "amendment"
    EXHIBIT = "exhibit"
    OTHER_RELATED = "other_related"


class RightsStatus(StrEnum):
    """Review status for using and redistributing a source."""

    PUBLIC_DOMAIN = "public_domain"
    OPEN_LICENSE = "open_license"
    PERMISSION_GRANTED = "permission_granted"
    REVIEW_REQUIRED = "review_required"
    RESTRICTED = "restricted"


class PartyRole(StrEnum):
    """Roles a named party may have in a lease."""

    LANDLORD = "landlord"
    TENANT = "tenant"
    GUARANTOR = "guarantor"
    PROPERTY_MANAGER = "property_manager"
    OTHER = "other"


class StakeholderPerspective(StrEnum):
    """Perspective requested for an explanation."""

    LANDLORD = "landlord"
    TENANT = "tenant"
    NEUTRAL = "neutral"


class AnswerStatus(StrEnum):
    """Evidence status for a document-grounded answer."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNREADABLE_SOURCE = "unreadable_source"


class TimingType(StrEnum):
    """How an obligation's timing is expressed."""

    FIXED_DATE = "fixed_date"
    RELATIVE_DEADLINE = "relative_deadline"
    RECURRING = "recurring"
    EVENT_BASED = "event_based"
    UNSTATED = "unstated"


class TimingUnit(StrEnum):
    """Normalized unit for a relative obligation deadline."""

    DAY = "day"
    BUSINESS_DAY = "business_day"
    MONTH = "month"
    YEAR = "year"


class AmendmentAction(StrEnum):
    """How an amendment changes an earlier document."""

    ADD = "add"
    DELETE = "delete"
    REPLACE = "replace"
    MODIFY = "modify"
    CONFIRM = "confirm"
    OTHER = "other"


class ClauseType(StrEnum):
    """Clause labels supported by ontology version 1."""

    PARTIES = "parties"
    PREMISES = "premises"
    PROPERTY_USE = "property_use"
    DEFINITIONS = "definitions"
    LEASE_TERM = "lease_term"
    COMMENCEMENT = "commencement"
    EXPIRATION = "expiration"
    RENEWAL_OPTION = "renewal_option"
    EXTENSION_OPTION = "extension_option"
    EARLY_TERMINATION = "early_termination"
    BASE_RENT = "base_rent"
    ADDITIONAL_RENT = "additional_rent"
    RENT_INCREASE = "rent_increase"
    RENT_ABATEMENT = "rent_abatement"
    TENANT_SHARE = "tenant_share"
    SECURITY_DEPOSIT = "security_deposit"
    OPERATING_EXPENSES = "operating_expenses"
    COMMON_AREA_MAINTENANCE = "common_area_maintenance"
    TAXES = "taxes"
    INSURANCE = "insurance"
    LATE_FEES = "late_fees"
    PERCENTAGE_RENT = "percentage_rent"
    MAINTENANCE = "maintenance"
    REPAIRS = "repairs"
    UTILITIES = "utilities"
    ALTERATIONS = "alterations"
    TENANT_IMPROVEMENTS = "tenant_improvements"
    DELIVERY_CONDITION = "delivery_condition"
    ACCESS = "access"
    COMMON_AREAS = "common_areas"
    SIGNAGE = "signage"
    PARKING = "parking"
    CONTINUOUS_OPERATION = "continuous_operation"
    CO_TENANCY = "co_tenancy"
    EXCLUSIVITY = "exclusivity"
    RADIUS_RESTRICTION = "radius_restriction"
    PROHIBITED_USE = "prohibited_use"
    RELOCATION = "relocation"
    ASSIGNMENT = "assignment"
    SUBLETTING = "subletting"
    CHANGE_OF_CONTROL = "change_of_control"
    AMENDMENT = "amendment"
    NOTICE = "notice"
    HOLDOVER = "holdover"
    SURRENDER = "surrender"
    SUBORDINATION = "subordination"
    NON_DISTURBANCE = "non_disturbance"
    ESTOPPEL = "estoppel"
    LIENS = "liens"
    BROKERAGE = "brokerage"
    DEFAULT = "default"
    CURE_PERIOD = "cure_period"
    REMEDIES = "remedies"
    INDEMNITY = "indemnity"
    LIABILITY = "liability"
    DAMAGE_AND_DESTRUCTION = "damage_and_destruction"
    CONDEMNATION = "condemnation"
    FORCE_MAJEURE = "force_majeure"
    ACCESSIBILITY = "accessibility"
    ENVIRONMENTAL = "environmental"
    COMPLIANCE_WITH_LAWS = "compliance_with_laws"
    SERVICES = "services"
    JANITORIAL = "janitorial"
    SECURITY = "security"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GOVERNING_LAW = "governing_law"
    GUARANTEE = "guarantee"
    EXHIBIT = "exhibit"
    SCHEDULE = "schedule"
    OTHER = "other"
