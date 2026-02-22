"""
Frontend QA tests using Streamlit's built-in AppTest framework.

Tests the dashboard headlessly — no browser needed. Catches:
- Runtime exceptions in any view
- Widget rendering failures
- Data loading issues
- View switching (selectbox, tabs, expanders)

Run: .venv/bin/pytest tests/test_frontend.py -v
"""

import pytest
from streamlit.testing.v1 import AppTest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app() -> AppTest:
    """Load and run the dashboard app once for all tests in this module."""
    at = AppTest.from_file("dashboard_app.py", default_timeout=30)
    at.run()
    return at


# ---------------------------------------------------------------------------
# ACT 0: App Loads
# ---------------------------------------------------------------------------

class TestAppLoads:
    """Verify the app boots without crashing."""

    def test_no_exception_on_load(self, app: AppTest) -> None:
        """App should load without any uncaught exceptions."""
        assert not app.exception, (
            f"App raised an exception on load: {app.exception}"
        )

    def test_no_error_messages(self, app: AppTest) -> None:
        """App should not show any st.error() messages."""
        errors = [e.value for e in app.error]
        assert len(errors) == 0, f"App displayed error(s): {errors}"

    def test_has_markdown_content(self, app: AppTest) -> None:
        """App should render at least some markdown (hero title, section headers)."""
        assert len(app.markdown) > 0, "No markdown content rendered"


# ---------------------------------------------------------------------------
# ACT 1: Executive View (Hero)
# ---------------------------------------------------------------------------

class TestExecutiveView:
    """Verify ACT 1 renders without errors."""

    def test_no_exception(self, app: AppTest) -> None:
        """Executive view should render without exceptions."""
        assert not app.exception, f"Executive view exception: {app.exception}"


# ---------------------------------------------------------------------------
# ACT 3: Selectbox + View Switching
# ---------------------------------------------------------------------------

class TestAct3ViewSwitching:
    """Verify the ACT 3 selectbox switches between Model Performance and Drift Monitor."""

    def test_selectbox_exists(self, app: AppTest) -> None:
        """The Analysis View selectbox should be present."""
        assert len(app.selectbox) > 0, "No selectbox found in the app"

    def test_default_is_model_performance(self, app: AppTest) -> None:
        """Default selectbox value should be 'Model Performance'."""
        selectbox = app.selectbox[0]
        assert selectbox.value == "Model Performance", (
            f"Expected default 'Model Performance', got '{selectbox.value}'"
        )

    def test_model_performance_has_tabs(self, app: AppTest) -> None:
        """Model Performance view should have Overfitting + Segment tabs."""
        assert len(app.tabs) > 0, "No tabs rendered in Model Performance view"

    def test_switch_to_drift_monitor(self, app: AppTest) -> None:
        """Switching to Drift Monitor should not raise exceptions."""
        at = AppTest.from_file("dashboard_app.py", default_timeout=60)
        at.run()
        assert not at.exception, f"Initial load exception: {at.exception}"

        # Switch selectbox
        at.selectbox[0].set_value("Drift Monitor")
        at.run()

        assert not at.exception, (
            f"Exception after switching to Drift Monitor: {at.exception}"
        )

    def test_switch_back_to_model_performance(self, app: AppTest) -> None:
        """Switching back to Model Performance should not raise exceptions."""
        at = AppTest.from_file("dashboard_app.py", default_timeout=60)
        at.run()

        # Switch to Drift Monitor
        at.selectbox[0].set_value("Drift Monitor")
        at.run()
        assert not at.exception

        # Switch back
        at.selectbox[0].set_value("Model Performance")
        at.run()
        assert not at.exception


# ---------------------------------------------------------------------------
# ACT 3: Drift Monitor View
# ---------------------------------------------------------------------------

class TestDriftMonitorView:
    """Verify the Drift Monitor view renders correctly."""

    @pytest.fixture()
    def drift_app(self) -> AppTest:
        """Load app and switch to Drift Monitor."""
        at = AppTest.from_file("dashboard_app.py", default_timeout=60)
        at.run()
        at.selectbox[0].set_value("Drift Monitor")
        at.run()
        return at

    def test_no_exception(self, drift_app: AppTest) -> None:
        """Drift Monitor should render without exceptions."""
        assert not drift_app.exception, (
            f"Drift Monitor raised: {drift_app.exception}"
        )

    def test_no_errors(self, drift_app: AppTest) -> None:
        """Drift Monitor should not display any st.error()."""
        errors = [e.value for e in drift_app.error]
        assert len(errors) == 0, f"Drift Monitor errors: {errors}"

    def test_has_markdown_content(self, drift_app: AppTest) -> None:
        """Drift Monitor should render content (status badge, table, actions)."""
        assert len(drift_app.markdown) > 0, "No content rendered in Drift Monitor"


# ---------------------------------------------------------------------------
# ACT 3: Segments View — Radio Button
# ---------------------------------------------------------------------------

class TestSegmentsView:
    """Verify segment view radio button works."""

    def test_radio_exists(self, app: AppTest) -> None:
        """Segment view should have a radio button (Trade/Line)."""
        assert len(app.radio) > 0, "No radio button found"

    def test_switch_radio_to_line(self, app: AppTest) -> None:
        """Switching radio to 'Line' should not raise exceptions."""
        at = AppTest.from_file("dashboard_app.py", default_timeout=30)
        at.run()
        assert not at.exception

        if len(at.radio) > 0:
            at.radio[0].set_value("Line")
            at.run()
            assert not at.exception, (
                f"Exception after switching radio to Line: {at.exception}"
            )


# ---------------------------------------------------------------------------
# ACT 4: Graveyard Expander
# ---------------------------------------------------------------------------

class TestGraveyardView:
    """Verify the graveyard expander renders."""

    def test_expander_exists(self, app: AppTest) -> None:
        """The graveyard expander should be present."""
        assert len(app.expander) > 0, "No expander found (graveyard missing)"

    def test_no_exception_with_expander(self, app: AppTest) -> None:
        """App with graveyard expander should have no exceptions."""
        assert not app.exception
