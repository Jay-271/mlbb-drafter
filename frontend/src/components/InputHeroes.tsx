import React from "react";

interface Props {
  section: string
  titles: string[];
  items: string[];
  selectedHeroes: string[];
  setSelectedHeroes: React.Dispatch<React.SetStateAction<string[]>>;
}

function InputHeroes({ section, titles, items, selectedHeroes, setSelectedHeroes }: Props) {
  const handleSelectionChange = (index: number, value: string) => {
    const updatedHeroes = [...selectedHeroes];
    updatedHeroes[index] = value; // Update the hero at the selected index
    setSelectedHeroes(updatedHeroes); // Update the parent state
    
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap" }}>
      {titles.map((title, index) => (
        <div key={index}>
          <select
            value={selectedHeroes[index]} // Controlled component
            onChange={(e) => handleSelectionChange(index, e.target.value)}
            id={section}
          >
            <option value="">{title}</option>
            {items.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
      ))}
    </div>
  );
}

export default InputHeroes;
