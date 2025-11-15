import torch
from torch.amp import autocast, GradScaler
from src.model.losses import combined_loss

def train_model(model, train_loader, val_loader, epochs=10, accum_steps=4, lr=1e-4, device="cuda"):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda")

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (X, Y) in enumerate(train_loader):
            X = X.float().to(device)
            Y = Y.long().to(device)

            with autocast("cuda"):
                pred = model(X)
                loss = combined_loss(pred, Y) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast("cuda"):
            for X, Y in val_loader:
                X = X.float().to(device)
                Y = Y.long().to(device)
                pred = model(X)
                loss = combined_loss(pred, Y)
                val_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
