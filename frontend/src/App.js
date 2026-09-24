import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    job_title: "Data Scientist",
    experience: 3,
    location: "Bangalore",
    company: "ABC Technologies",
    company_size: "Large",
    industry: "Technology",
    education: "Bachelor's",
    skills: ["Python", "ML", "SQL"],
    posted_date: new Date().toISOString().split("T")[0],
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: name === "experience" ? Number(value) : value,
    });
  };

  const handleSkillsChange = (e) => {
    const skills = e.target.value
      .split(",")
      .map((skill) => skill.trim())
      .filter((skill) => skill !== "");

    setFormData({
      ...formData,
      skills,
    });
  };

  const handlePredict = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await axios.post(
        "http://localhost:8000/predict",
        formData
      );

      setResult(response.data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to make prediction. Please make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">

      {/* NAVBAR */}

      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">₹</div>

          <div>
            <h2>SalaryAI</h2>
            <span>ML Salary Estimator</span>
          </div>
        </div>

        <div className="nav-status">
          <span className="status-dot"></span>
          Model Online
        </div>
      </nav>

      {/* HERO */}

      <section className="hero">
        <div className="hero-content">
          <span className="hero-badge">
            AI-POWERED SALARY PREDICTION
          </span>

          <h1>
            Know your
            <span> market value.</span>
          </h1>

          <p>
            Estimate your expected salary using machine learning
            based on your experience, skills, location and profile.
          </p>
        </div>
      </section>

      {/* MAIN */}

      <main className="dashboard">

        {/* PROFILE CARD */}

        <section className="profile-card">

          <div className="section-heading">
            <div className="heading-icon">👤</div>

            <div>
              <h2>Your Profile</h2>
              <p>Enter your professional information</p>
            </div>
          </div>

          <form onSubmit={handlePredict}>

            <div className="form-grid">

              <div className="form-group full-width">
                <label>Job Title</label>

                <input
                  type="text"
                  name="job_title"
                  value={formData.job_title}
                  onChange={handleChange}
                  placeholder="e.g. Data Scientist"
                  required
                />
              </div>

              <div className="form-group">
                <label>Experience</label>

                <div className="input-with-unit">
                  <input
                    type="number"
                    name="experience"
                    min="0"
                    max="50"
                    value={formData.experience}
                    onChange={handleChange}
                    required
                  />

                  <span>years</span>
                </div>
              </div>

              <div className="form-group">
                <label>Location</label>

                <select
                  name="location"
                  value={formData.location}
                  onChange={handleChange}
                >
                  <option>Bangalore</option>
                  <option>Hyderabad</option>
                  <option>Delhi</option>
                  <option>Mumbai</option>
                  <option>Pune</option>
                  <option>Chennai</option>
                  <option>Kolkata</option>
                  <option>Gurugram</option>
                  <option>Noida</option>
                  <option>Jaipur</option>
                </select>
              </div>

              <div className="form-group">
                <label>Company</label>

                <input
                  type="text"
                  name="company"
                  value={formData.company}
                  onChange={handleChange}
                  placeholder="Company name"
                  required
                />
              </div>

              <div className="form-group">
                <label>Company Size</label>

                <select
                  name="company_size"
                  value={formData.company_size}
                  onChange={handleChange}
                >
                  <option>Startup</option>
                  <option>Small</option>
                  <option>Mid</option>
                  <option>Large</option>
                  <option>Enterprise</option>
                </select>
              </div>

              <div className="form-group">
                <label>Industry</label>

                <input
                  type="text"
                  name="industry"
                  value={formData.industry}
                  onChange={handleChange}
                  placeholder="e.g. Technology"
                  required
                />
              </div>

              <div className="form-group">
                <label>Education</label>

                <select
                  name="education"
                  value={formData.education}
                  onChange={handleChange}
                >
                  <option>Bachelor's</option>
                  <option>Master's</option>
                  <option>PhD</option>
                  <option>Other</option>
                </select>
              </div>

              <div className="form-group full-width">
                <label>Skills</label>

                <input
                  type="text"
                  value={formData.skills.join(", ")}
                  onChange={handleSkillsChange}
                  placeholder="Python, ML, SQL"
                  required
                />

                <small>
                  Separate multiple skills using commas
                </small>
              </div>

            </div>

            <button
              type="submit"
              className="predict-btn"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Predicting...
                </>
              ) : (
                <>
                  🚀 Predict My Salary
                </>
              )}
            </button>

          </form>

          {error && (
            <div className="error-message">
              ❌ {error}
            </div>
          )}

        </section>

        {/* RESULT CARD */}

        <section className="result-card">

          {!result ? (
            <div className="empty-result">

              <div className="empty-icon">
                💰
              </div>

              <h2>Your Salary Estimate</h2>

              <p>
                Fill in your profile and click
                <strong> Predict My Salary </strong>
                to see your estimated market value.
              </p>

              <div className="empty-features">

                <div>
                  <span>📊</span>
                  <p>ML Prediction</p>
                </div>

                <div>
                  <span>💼</span>
                  <p>Profile Analysis</p>
                </div>

                <div>
                  <span>📈</span>
                  <p>Salary Range</p>
                </div>

              </div>

            </div>
          ) : (
            <div className="prediction-result">

              <div className="result-heading">
                <div>
                  <span className="result-label">
                    ESTIMATED SALARY
                  </span>

                  <h2>Your Market Estimate</h2>
                </div>

                <span className="result-check">
                  ✓
                </span>
              </div>

              <div className="salary-value">
                <span>₹</span>
                {result.predicted_salary}
                <small>L</small>
              </div>

              <p className="salary-caption">
                Estimated annual salary
              </p>

              <div className="range-container">

                <div className="range-box">
                  <span>Minimum</span>
                  <strong>
                    ₹{result.salary_range_min} L
                  </strong>
                </div>

                <div className="range-line"></div>

                <div className="range-box">
                  <span>Maximum</span>
                  <strong>
                    ₹{result.salary_range_max} L
                  </strong>
                </div>

              </div>

              <div className="confidence-box">

                <div>
                  <span className="confidence-title">
                    Prediction Confidence
                  </span>

                  <span className="confidence-description">
                    Based on your profile
                  </span>
                </div>

                <span
                  className={`confidence-badge ${result.confidence.toLowerCase()}`}
                >
                  {result.confidence}
                </span>

              </div>

              <div className="profile-summary">

                <h3>Profile Summary</h3>

                <div className="summary-grid">

                  <div>
                    <span>Job Title</span>
                    <strong>{formData.job_title}</strong>
                  </div>

                  <div>
                    <span>Experience</span>
                    <strong>
                      {formData.experience} years
                    </strong>
                  </div>

                  <div>
                    <span>Location</span>
                    <strong>{formData.location}</strong>
                  </div>

                  <div>
                    <span>Skills</span>
                    <strong>
                      {formData.skills.length} skills
                    </strong>
                  </div>

                </div>

              </div>

            </div>
          )}

        </section>

      </main>

      {/* FOOTER */}

      <footer className="footer">
        <p>
          SalaryAI • Built with React + FastAPI + Machine Learning
        </p>
      </footer>

    </div>
  );
}

export default App;
