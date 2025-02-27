import { useState, useEffect } from "react";
import InputHeroes from "../../components/InputHeroes";
import { Button, Spinner } from "react-bootstrap";
import axios from "axios";
import "../../assets/middle.css";

interface Props {
  baseAPILink: string;
}

function Middle({ baseAPILink }: Props) {
  const [ourTeam, setOurTeam] = useState(Array(5).fill(""));
  const [enemyTeam, setEnemyTeam] = useState(Array(5).fill(""));
  const [predFull, setPredFull] = useState(false);
  const [bannedHeroes, setBannedHeroes] = useState(Array(10).fill(""));
  const [sampleHeroes, setSampleHeroes] = useState<string[]>([]);
  const [ourPickImage, setOurPickImage] = useState("");
  const [enemyPickImage, setEnemyPickImage] = useState("");
  const [loading, setLoading] = useState(false); // For showing loading spinner
  const [heroLoading, setHeroLoading] = useState(false); // For showing when backend ready
  const [loadingTraining, setLoadingTraining] = useState(false); // For showing loading spinner
  const [highlightedImage, setHighlightedImage] = useState<string | null>(null); // For highlighting the image.
  const [highlightedEnemyImage, setEnemyHighlightedImage] = useState<
    string | null
  >(null); // For highlighting the image.
  const [topN, setTopN] = useState<number>(7);
  const [targetPick, setTargetPick] = useState(Array(1).fill(""));
  const [trainingResponse, setTrainingResponse] = useState<string>();

  const titles = ["1ˢᵗ hero", "2ⁿᵈ hero", "3ʳᵈ hero", "4ᵗʰ hero", "5ᵗʰ hero"];
  const bannedTitles = [
    "1ˢᵗ hero",
    "2ⁿᵈ hero",
    "3ʳᵈ hero",
    "4ᵗʰ hero",
    "5ᵗʰ hero",
    "6ᵗʰ hero",
    "7ᵗʰ hero",
    "8ᵗʰ hero",
    "9ᵗʰ hero",
    "10ᵗʰ hero",
  ];
  const trainingTitle = ["Please choose the correct hero."];

  // Fetch heroes on mount
  useEffect(() => {
    const fetchHeroes = async () => {
      setHeroLoading(true); // Show loading something
      try {
        const response = await axios.get(baseAPILink + "/api/heroes", {
          headers: {
            "ngrok-skip-browser-warning": "123",
          },
        });
        setSampleHeroes(response.data.message); // Update original hero array
      } catch (e) {
        console.error("Error getting heroes: ", e);
        setSampleHeroes([""]); // Fallback if there is an error, empty arr will show no heroes.
      } finally {
        setHeroLoading(false); // Hide loading something
      }
    };
    fetchHeroes();
  }, []); // Empty dependency array ensures this runs only once

  const fetchRecommendations = async () => {
    setLoading(true); // Show loading spinner (for logic below)
    if (!ourTeam.includes("") && !enemyTeam.includes("")) {
      if (!predFull) {
        setPredFull(true);
        setLoading(false);
        throw Error("Draft is already finalized.");
      }
    } else {
      setPredFull(false);
    }

    try {
      const response = await axios.post(
        baseAPILink + "/api/predictions",
        {
          our_picks: ourTeam,
          enemy_picks: enemyTeam,
          banned_heroes: bannedHeroes,
          top_n: topN,
        },
        {
          headers: {
            "ngrok-skip-browser-warning": "123",
          },
        }
      );
      setOurPickImage(response.data.ourPick);
      setEnemyPickImage(response.data.enemyPick);
      setHighlightedImage("ourPickImage"); // Trigger highlight effect
      setEnemyHighlightedImage("enemyPickImage");
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    } finally {
      setLoading(false); // Hide loading spinner
    }
  };

  const handleUserTopN = (event: React.KeyboardEvent<HTMLInputElement>) => {
    const inputValue = parseInt((event.target as HTMLInputElement).value, 10);
    if (!isNaN(inputValue) && inputValue < 101) {
      setTopN(inputValue); // Update state if the input is a valid number
      console.log("Updated recommendations number:", inputValue);
    } else {
      console.warn(
        "Please input a valid number. Value will be what it was previously..."
      );
    }
  };

  const updateModels = async () => {
    setLoadingTraining(true); // Show loading spinner (for logic below)
    try {
      const response = await axios.post(
        baseAPILink + "/api/feedback",
        {
          our_picks: ourTeam,
          enemy_picks: enemyTeam,
          banned_heroes: bannedHeroes,
          target_pick: targetPick,
        },
        {
          headers: {
            "ngrok-skip-browser-warning": "123",
          },
        }
      );
      setTrainingResponse(response.data.message);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    } finally {
      setLoadingTraining(false); // Hide loading spinner
    }
  };

  const resetHeroes = async () => {
    setBannedHeroes(Array(10).fill(""));
    setEnemyTeam(Array(5).fill(""));
    setOurTeam(Array(5).fill(""));
  };

  return (
    <>
      <div className="logoImg"></div>
      <h1 id="main-title">--- Welcome to the World's First ---</h1>
      <h2 id="main-subtitle">** Mobile Legends: Bang Bang AI-Drafter! **</h2>
      <h6 id="main-author">// By Jay :) \\</h6>
      <p>
        Note* AI Models are <strong>big!</strong> <br />
        <strong>Please</strong> give this page a <strong>couple seconds</strong>{" "}
        to <strong>load</strong> all the models in the <strong>backend</strong>.{" "}
        <br />
        ++ Happy drafting ++ <br /> <br />
        {/* use the loading something */}
        {heroLoading ? (
          <div>
            Loading models...
            <Spinner
              animation="border"
              role="status"
              style={{ margin: "auto", display: "block" }}
            >
              <span className="sr-only"></span>
            </Spinner>
          </div>
        ) : (
          <span style={{fontSize: "2rem" }}>✔️</span>
        )}
      </p>
      <div className="picksDiv">
        <h4 id="Banned">Banned Heroes</h4>
        <span>(These won't show up during predictions)</span>
        <InputHeroes
          section="bannedHeroes"
          titles={bannedTitles}
          items={sampleHeroes}
          selectedHeroes={bannedHeroes}
          setSelectedHeroes={setBannedHeroes}
        />
      </div>

      <div className="picksDiv">
        <h4 id="Team">Blue Side Picks:</h4>
        <InputHeroes
          section="blueSideHeroes"
          titles={titles}
          items={sampleHeroes}
          selectedHeroes={ourTeam}
          setSelectedHeroes={setOurTeam}
        />
      </div>
      <div className="picksDiv">
        <h4 id="Enemy">Red Side Picks:</h4>
        <InputHeroes
          section="redSideHeroes"
          titles={titles}
          items={sampleHeroes}
          selectedHeroes={enemyTeam}
          setSelectedHeroes={setEnemyTeam}
        />
      </div>

      <h4>Please choose the number of recommendations:</h4>
      <form>
        <div
          className="form-group"
          style={{ textAlign: "center", margin: "2rem" }}
        >
          <label>
            Please input the number of recommendations. (A lower number will
            return fewer predictions, but performance remains the same...):
          </label>
          <div id="banInput">
            <input
              type="text"
              className="form-control"
              id="reccNum"
              placeholder="Type a number... e.g., 1, 2, 10, etc."
              onChangeCapture={handleUserTopN}
              defaultValue={topN} // Set the default value
            />
          </div>
        </div>
      </form>

      <div style={{ textAlign: "center", margin: "2rem" }}>
        <Button size="lg" variant="primary" onClick={fetchRecommendations}>
          {loading ? (
            <>
              <Spinner
                as="span"
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
              />
              Hold on a sec, we cooking chat...
            </>
          ) : (
            "Send it"
          )}
        </Button>{" "}
        <br /> <br />
        <Button size="lg" variant="secondary" onClick={resetHeroes}>
          Reset heroes & Bans
        </Button>
      </div>
      <div style={{ textAlign: "center", margin: "2rem" }}>
        {predFull ? (
          <h1 style={{ fontSize: "100px" }}>
            Draft is already complete. <br />
            <h2>Start a new draft to continue.</h2>
          </h1>
        ) : (
          <>
            <h5>
              When drafting for your team, remember to click the update button
              after selecting each hero. This will provide you with a new
              recommendation based on the current synergy with the heroes you've
              already chosen.
            </h5>
            {ourPickImage && (
              <>
                <h2 style={{ marginTop: "10px", color: "blue" }}>
                  <ul>The Next Hero Blue Side Should Pick:</ul>
                </h2>
                <img
                  src={`data:image/png;base64,${ourPickImage}`}
                  alt="Our Picks"
                  className={
                    highlightedImage === "ourPickImage" ? "highlight" : ""
                  }
                />
              </>
            )}
            {enemyPickImage && (
              <>
                <h2 style={{ marginTop: "10px", color: "red" }}>
                  <ul>The Next Hero Red Side Should Pick:</ul>
                </h2>
                <img
                  src={`data:image/png;base64,${enemyPickImage}`}
                  alt="Enemy Picks"
                  className={
                    highlightedEnemyImage === "enemyPickImage"
                      ? "highlight"
                      : ""
                  }
                />
              </>
            )}
            <div className="Training">
              {highlightedEnemyImage && (
                <>
                  <h1 id="training-header">
                    Want to help train our models? If so do it here!
                  </h1>

                  <p>
                    This action will send the currently selected draft at the
                    top to the server for training. The target value (dropdown
                    button) should represent the expected top recommendation
                    (based on the images above).
                  </p>
                  <InputHeroes
                    section="trainingHeroes"
                    titles={trainingTitle}
                    items={sampleHeroes}
                    selectedHeroes={targetPick}
                    setSelectedHeroes={setTargetPick}
                  />
                  <p>
                    For example: If the Blue side picked Atlas and after running
                    the models, the server suggests Thamuz as the next best pick
                    for the Red side, the target could be a pick like Diggie, or
                    another reasonable option, as it directly counters the Atlas
                    pick.
                  </p>

                  <Button
                    id="trainButton"
                    size="lg"
                    variant="danger"
                    onClick={updateModels}
                  >
                    {loadingTraining ? (
                      <>
                        <Spinner
                          as="span"
                          animation="border"
                          size="sm"
                          role="status"
                          aria-hidden="true"
                        />
                        Chotto matte, training models...
                      </>
                    ) : (
                      "Train Models!"
                    )}
                  </Button>
                </>
              )}
              {trainingResponse && <p>{trainingResponse}</p>}
            </div>
          </>
        )}
      </div>
    </>
  );
}

export default Middle;
